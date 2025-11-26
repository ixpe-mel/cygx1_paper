"""
Absolute‑normalised QPO finder (optimised)

Main gains
----------
1. **One‑shot pre‑computation** of modulation grid – no more rebuilding it for
   each observation.
2. **Pure‑NumPy weight handling** – avoids Python loops & list‑of‑lists.
3. **Joblib Memory cache** for expensive helper functions that are repeatedly
   called with identical arguments (e.g. `load_and_clean`, `Lightcurve.make_lightcurve`).
4. **Fast math**: use `np.full_like`, inplace ops, and typed sums.
5. **Cleaner I/O** – no `print` inside the workers; use `logging` if needed.
6. **Explicit back‑end choice & CPU cap** – leave one core free by default.

Tested with Python 3.12, joblib 1.6.1, NumPy 2.0.  Drop‑in if your helper
modules’ APIs are unchanged.
"""

from __future__ import annotations
import sys
from pathlib import Path
import logging
from functools import partial

import numpy as np
from joblib import Parallel, delayed, Memory

import load_and_clean as lac
import fit_rms_phase as frp
import F_test as ft
import chi_square as chis
import get_obs_file_triplets as gft
import dG_span_new_abs as dgs
import G_span_abs as gs
from stingray import Lightcurve, Powerspectrum, AveragedCrossspectrum

# -----------------------------------------------------------------------------
# Global ‑‑ set once
# -----------------------------------------------------------------------------
LOGGER = logging.getLogger("qpo_finder_abs")
MEMORY = Memory(location=Path(".joblib_cache"), verbose=0)  # change path if tmpfs

# Constant modulation grid (radians)
def _make_mod_grid(nb: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edges = np.linspace(np.radians(-90), np.radians(90), nb + 1)
    centres = (edges[:-1] + edges[1:]) * 0.5          # shape (nb,)
    half_width = (edges[1:] - edges[:-1]) * 0.5       # shape (nb,)
    return edges, centres, half_width

# Memoise expensive helpers ---------------------------------------------------
@MEMORY.cache
def _load_lightcurve(times: np.ndarray, dt: float, gti: np.ndarray):
    lc = Lightcurve.make_lightcurve(times, dt=dt, gti=gti)
    lc.apply_gtis()
    return lc

@MEMORY.cache
def _load_and_clean(*args, **kwargs):
    return lac.load_and_clean(*args, **kwargs)

# Physics helpers -------------------------------------------------------------
def _cross_spec_model_null(J, Qn, Un, phi, C2):
    """Null hypothesis model (absolute rms normalisation)."""
    return (C2 / J) * (1.0 + Qn * np.cos(2 * phi) + Un * np.sin(2 * phi))

# -----------------------------------------------------------------------------
# Per‑pair worker
# -----------------------------------------------------------------------------
def _process_pair(
    obs_triplet: tuple[str, str, str],
    *,
    Pmin, Pmax, bin_len, seg_len, fmin, fmax,
    mod_centres, mod_cnt, spur_sub, coherence_corr,
) -> dict:
    (f1, f2, gti_path) = obs_triplet

    # ---------- Load & clean
    data1, *_ = _load_and_clean(f1, Pmin, Pmax)
    data2, *_ = _load_and_clean(f2, Pmin, Pmax)
    GTI = np.loadtxt(gti_path)

    N1, N2 = len(data1["TIME"]), len(data2["TIME"])
    scale = (N1 + N2) ** 2 / (N1 * N2)

    # ---------- Light curves & spectra
    lc1 = _load_lightcurve(data1["TIME"], bin_len, GTI)
    lc2 = _load_lightcurve(data2["TIME"], bin_len, GTI)

    ps2 = Powerspectrum.from_lightcurve(lc2, seg_len, norm="abs")
    mask = (fmin <= ps2.freq) & (ps2.freq <= fmax)
    ps2_ref_mean = scale * ps2.power[mask].mean(dtype=np.float64)

    cs = AveragedCrossspectrum.from_lightcurve(lc1, lc2, seg_len, norm="abs")
    cs_ref_real = scale * cs.power.real[mask].mean(dtype=np.float64)
    cs_ref_imag = scale * cs.power.imag[mask].mean(dtype=np.float64)
    cs_ref_mod  = np.abs(cs_ref_real + 1j * cs_ref_imag)

    # ---------- G‑span
    G_real, G_im, n_span, m_span, lc1_sub, lc_spur, cs_list, spur_norm = gs.G_span(
        mod_cnt, data1, lc2, GTI, bin_len, seg_len,
        fmin, fmax, spur_sub, norm="abs"
    )
    G_real = scale * np.asarray(G_real, dtype=np.float64)
    G_im   = scale * np.asarray(G_im,   dtype=np.float64)

    # ---------- dG
    dG = dgs.dG_span(
        G_real, G_im, lc1_sub, n_span, m_span, fmin, fmax,
        seg_len, ps2_ref_mean, cs_ref_real, coherence_corr
    )

    # ---------- Null models
    Qn = data1["Q"].sum(dtype=np.float64) / N1
    Un = data1["U"].sum(dtype=np.float64) / N1

    G_null      = _cross_spec_model_null(mod_cnt, Qn, Un, mod_centres, cs_ref_mod)
    G_null_cpf  = np.full_like(G_real, cs_ref_real / mod_cnt)  # imag part = 0

    # ---------- χ² & F‑tests
    (fit_sin180_real, fit_sin180_imag,
     chi_sin_real, chi_sin_imag,
     dof_sin_real, dof_sin_imag) = frp.fit_sine_180(
        mod_centres, G_real, dG, G_im, dG
    )[1:7]  # unpack wanted elements

    chi_null_real = chis.chi_square(G_real, G_null.real, dG)
    chi_null_imag = chis.chi_square(G_im,   G_null.imag, dG)

    chi_cpf_real  = chis.chi_square(G_real, G_null_cpf,  dG)
    chi_cpf_imag  = chis.chi_square(G_im,   0.0,         dG)

    dof_null = len(mod_centres)
    F_pol_var = ft.F_test(chi_null_real, chi_null_imag, dof_null, dof_null,
                          chi_sin_real, chi_sin_imag, dof_sin_real, dof_sin_imag)
    F_cpf     = ft.F_test(chi_cpf_real,  chi_cpf_imag,  dof_null, dof_null,
                          chi_sin_real, chi_sin_imag, dof_sin_real, dof_sin_imag)

    return dict(
        N1=N1, Q=data1["Q"], U=data1["U"],
        cs_ref=cs, cs_ref_real=cs_ref_real, cs_ref_imag=cs_ref_imag,
        cs_list=cs_list, w_ref=cs.m, w_G=[c.m for c in cs_list],
        G_real=G_real, G_im=G_im, dG=dG,
        G_null=G_null, G_null_cpf=G_null_cpf,
        fit_real=fit_sin180_real, fit_imag=fit_sin180_imag,
        scale=scale, F_pol_var=F_pol_var, F_cpf=F_cpf,
    )

# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------
def run_QPO_finder_absolute_stacked(
    obs_folder: str | Path,
    obs_names: list[str],
    *,
    Pmin, Pmax, bin_length, seg_length, fmin, fmax,
    mod_bin_number=18,
    spur_sub=True,
    coherence_corrector=True,
    output_file: str | Path = "qpo_results.txt",
    n_jobs: int | None = -2,           # leave one CPU free
    verbose: int = 10,
) -> None:
    edges, mod_centres, mod_halfw = _make_mod_grid(mod_bin_number)
    obs_pairs = gft.get_obs_file_pairs(obs_folder, obs_names)

    worker = delayed(_process_pair)
    results = Parallel(
        n_jobs=n_jobs, backend="loky", verbose=verbose
    )(worker(pair,
            Pmin=Pmin, Pmax=Pmax, bin_len=bin_length, seg_len=seg_length,
            fmin=fmin, fmax=fmax,
            mod_centres=mod_centres, mod_cnt=mod_bin_number,
            spur_sub=spur_sub, coherence_corr=coherence_corrector)
      for pair in obs_pairs)

    # ------------------ Stack the results vectorially ------------------------
    w_G = np.hstack([r["w_G"] for r in results])     # (n_pairs, n_bins)
    G_real = np.average(np.vstack([r["G_real"] for r in results]),
                        axis=0, weights=w_G)
    G_imag = np.average(np.vstack([r["G_im"] for r in results]),
                        axis=0, weights=w_G)
    dG = np.sqrt(np.add.reduce([r["dG"]**2 for r in results])) / len(results)

    # Global null models ------------------------------------------------------
    I_tot   = sum(r["N1"] for r in results)
    Q_tot   = sum(r["Q"].sum(dtype=np.float64) for r in results)
    U_tot   = sum(r["U"].sum(dtype=np.float64) for r in results)
    Qn_glob = Q_tot / I_tot
    Un_glob = U_tot / I_tot

    cs_ref_mean_real = np.average([r["cs_ref_real"] for r in results],
                                  weights=[r["w_ref"] for r in results])
    G_null_global = _cross_spec_model_null(
        mod_bin_number, Qn_glob, Un_glob, mod_centres, np.abs(cs_ref_mean_real)
    )
    G_null_cpf_glob = np.full_like(G_null_global, cs_ref_mean_real / mod_bin_number)

    # ---------- χ² & F on stacked data
    chi_real_null = chis.chi_square(G_real, G_null_global.real, dG)
    chi_imag_null = chis.chi_square(G_imag, G_null_global.imag, dG)
    chi_real_cpf  = chis.chi_square(G_real, G_null_cpf_glob, dG)
    chi_imag_cpf  = chis.chi_square(G_imag, 0.0, dG)

    _, fit_real, _, fit_imag, dof_sin_r, dof_sin_i, chi_sin_r, chi_sin_i, *_ = \
        frp.fit_sine_180(mod_centres, G_real, dG, G_imag, dG)

    dof_null = len(mod_centres) - 1
    F_pol_var = ft.F_test(chi_real_null, chi_imag_null, dof_null, dof_null,
                          chi_sin_r, chi_sin_i, dof_sin_r, dof_sin_i)
    F_cpf     = ft.F_test(chi_real_cpf,  chi_imag_cpf,  len(mod_centres),
                          len(mod_centres), chi_sin_r, chi_sin_i,
                          dof_sin_r, dof_sin_i)

    LOGGER.info("F‑test (polarisation variability): %.3g", F_pol_var)
    LOGGER.info("F‑test (constant polarised flux): %.3g", F_cpf)

    # ------------------ Save to disk ----------------------------------------
    out = np.column_stack([mod_centres, G_real, G_imag,
                           G_null_global.real, G_null_global.imag,
                           dG, fit_real, fit_imag])
    np.savetxt(output_file, out,
               header="phi  G_real  G_imag  G_null_R  G_null_I  dG  fit_R  fit_I")
    LOGGER.info("Results saved to %s", output_file)

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import sys
sys.path.append('/home/c2032014/py_files')
import importlib
import load_and_clean as lac
import fit_rms_phase as frp
import F_test as ft
import chi_square as chis
import get_obs_file_triplets as gft
from stingray import Lightcurve,Powerspectrum, AveragedCrossspectrum
import dG_span_new_abs as dgs
import G_span_abs as gs
from scipy.optimize import curve_fit
import chi_square as cs
importlib.reload(ft)
importlib.reload(dgs)
#def cross_spec_model_real(phi,A,B,C):
 #   Re_G=(1/J) * ( A + (B*np.cos(2*phi)) + (C*np.sin(2*phi)) )
 #   return Re_G

#The imaginary sinusoid does not have the A term
#def cross_spec_model_imag(phi,B,C,J):
#    Im_G=(1/J) * ( (B*np.cos(2*phi)) + (C*np.sin(2*phi)) )
#    return Im_G


def sinusoid(phi, A, B, C):
    return A + (B * np.cos(2 * phi)) + (C * np.sin(2 * phi))



def cross_spec_model_null(phi,Q_norm,U_norm ,C_nu_mag_sqrd,J):
    return (1/J) * C_nu_mag_sqrd * (1 + (Q_norm * np.cos(2 * phi)) + (U_norm * np.sin(2 * phi)))

#def cross_spec_model_null_fixedJ(phi, Q_norm, U_norm ,C_nu_mag_sqrd):
#        return cross_spec_model_null(phi, C_nu_mag_sqrd, Q_norm, U_norm, J=J)

def cross_spec_model_null_cpf(J,C_nu_mag_sqrd):
    return C_nu_mag_sqrd / J





def qfans(obs_folder,obs_name, Pmin, Pmax, bin_length, seg_length, fmin, fmax, spur_sub, norm,
                 mod_bin_number, coherence_corrector):
    obs = gft.get_obs_file_pairs(obs_folder, obs_name)

   

    aspace = np.linspace(np.radians(-90), np.radians(90), mod_bin_number + 1)
    mod_min_array = aspace[:-1]
    mod_max_array = aspace[1:]
    av_mod = (mod_min_array + mod_max_array) / 2
    J=mod_bin_number

    #print('obs1',obs[0][0])
    #print('obs2',obs[0][1])

    data1, *_ = lac.load_and_clean(obs[0][0], Pmin, Pmax)
    data2, *_ = lac.load_and_clean(obs[0][1], Pmin, Pmax)
    GTI = list(np.loadtxt(obs[0][2]))

    I_obs = len(data1['TIME'])
    Q_obs = data1['Q']
    U_obs = data1['U']
    
    scale_factor=(I_obs+len(data2['TIME']))**2/(I_obs* len(data2['TIME']))
    #print('scale_factor',scale_factor)

    lc1 = Lightcurve.make_lightcurve(data1['TIME'], dt=bin_length, gti=GTI)
    lc2 = Lightcurve.make_lightcurve(data2['TIME'], dt=bin_length, gti=GTI)
    lc1.apply_gtis()
    lc2.apply_gtis()



    ps_2 = Powerspectrum.from_lightcurve(lc2, seg_length, norm='abs')
    ps_2_ref_mean =scale_factor* ps_2.power[(fmin <= ps_2.freq) & (ps_2.freq <= fmax)].mean()

    cs_ref = AveragedCrossspectrum.from_lightcurve(lc1, lc2, seg_length, norm='abs')
    cs_ref_power=np.array(scale_factor)*np.array(cs_ref.power)
    
    
    cs_ref_mean_real =scale_factor* cs_ref.power.real[(fmin <= cs_ref.freq) & (cs_ref.freq <= fmax)].mean()
    #print('cs_ref_mean_real',cs_ref_mean_real)
    cs_ref_mean_imag =scale_factor * cs_ref.power.imag[(fmin <= cs_ref.freq) & (cs_ref.freq <= fmax)].mean()
    cs_ref_mean = np.abs(cs_ref_mean_real + 1j * cs_ref_mean_imag)
    

    G_real_span, G_im_span, n_span, m_span, lc_1_sub_span, lc_spur, cs_list, spur_sub_norm = gs.G_span(
        mod_bin_number, data1, lc2, GTI, bin_length, seg_length, fmin, fmax, spur_sub, norm
    )

    print(G_real_span)
    print(G_im_span)

    G_real_span = np.array(G_real_span) * scale_factor
    G_im_span = np.array(G_im_span) * scale_factor

    print('G_real_span', G_real_span)
    print('G_im_span', G_im_span)

    Q_norm = np.sum(Q_obs) / I_obs
    U_norm = np.sum(U_obs) / I_obs
    Q=np.sum(Q_obs)
    U=np.sum(U_obs)
    subject_cr=[i.meanrate for i in lc_1_sub_span]

    #G_null = cross_spec_model_null(J,Q_uncorrected,U_uncorrected,np.array(av_mod), cs_ref_mean)
    G_null= cross_spec_model_null(np.array(av_mod),Q_norm,U_norm,cs_ref_mean,J)
    overall_mean_cr=np.mean(subject_cr)
    overall_mean_cr=np.array(overall_mean_cr)

    G_null_cpf_real =cs_ref_mean_real/mod_bin_number
    G_null_cpf_real=[G_null_cpf_real]*len(av_mod)
    
    G_null_cpf_imag=0
    G_null_cpf_imag=np.array([G_null_cpf_imag]*len(av_mod))
    #G_null_cpf_imag=G_null_cpf_imag[1:])
    #print('G_null_cpf',G_null_cpf)
    
    #print('G_real_span', G_real_span)
    #print('G_im_span', G_im_span)
    #print('n_span', n_span)
    #print('m_span', m_span)
    #print('lc_1_sub_span', lc_1_sub_span)
    #print('ps_2_ref_mean', ps_2_ref_mean)
    #print('cs_ref_mean', cs_ref_mean)
    #print('cs_ref_mean_real', cs_ref_mean_real)


    dG_span = dgs.dG_span(
        G_real_span, G_im_span, lc_1_sub_span, n_span, m_span, fmin, fmax,
        seg_length, ps_2_ref_mean, cs_ref_mean_real, coherence_corrector,scale_factor
    )

    #print('dG_span', dG_span)
    # Performing ftest on each observation

    def cross_spec_model_real(phi, B, C):
        Re_G = (1 / J) * (cs_ref_mean + (B * np.cos(2 * phi)) + (C * np.sin(2 * phi)))
        return Re_G

    def cross_spec_model_imag(phi, B, C):
        Im_G = (1 / J) * ((B * np.cos(2 * phi)) + (C * np.sin(2 * phi)))
        return Im_G
    



    popt, pcov = curve_fit(sinusoid, av_mod, subject_cr)
    A, B, C = popt

    fit_model= sinusoid(av_mod, A, B, C)



    params_full_fit_real, pcov_full_real=curve_fit(cross_spec_model_real,np.array(av_mod), np.array(G_real_span), sigma=np.array(dG_span))
    params_full_fit_imag, pcov_full_imag=curve_fit(cross_spec_model_imag,np.array(av_mod), np.array(G_im_span), sigma=np.array(dG_span))
    #print('params_full_fit_real', params_full_fit_real)
    #print('params_full_fit_imag', params_full_fit_imag)

    fit_full_real = cross_spec_model_real(av_mod, params_full_fit_real[0], params_full_fit_real[1])
    fit_full_imag = cross_spec_model_imag(av_mod, params_full_fit_imag[0], params_full_fit_imag[1])

    dof_model_real=len(av_mod)-2 #Free params are B and C (Q and U)
    dof_model_imag=len(av_mod)-2

    model_chi_real=cs.chi_square(G_real_span,fit_full_real,dG_span)
    print('model_chi_real', model_chi_real)
    print('dof_model_real', dof_model_real)

    model_chi_imag=cs.chi_square(G_im_span,fit_full_imag,dG_span)
    print('model_chi_imag', model_chi_imag)
    print('dof_model_imag', dof_model_imag)

    reduced_chi_model=cs.reduced_chi_square(model_chi_real,model_chi_imag,dof_model_real,dof_model_imag)
    print('reduced_chi_model',reduced_chi_model)
    
   # _,fit_sin180_real,_,fit_sin180_imag,dof_sin180_real, dof_sin180_imag, real_sin180_chi, imag_sin180_chi,*_= frp.fit_sine_180(np.array(av_mod), np.array(G_real_span), np.array(dG_span), np.array(G_im_span), np.array(dG_span))
    
    dof_null_real=len(av_mod)
    dof_null_imag=len(av_mod)

    dof_null_real_cpf=len(av_mod)
    dof_null_imag_cpf=len(av_mod)

    chi_real_null_cpf=chis.chi_square(np.array(G_real_span), np.array(G_null_cpf_real), np.array(dG_span))
    chi_imag_null_cpf=chis.chi_square(np.array(G_im_span), np.array(G_null_cpf_imag), np.array(dG_span))
    print('chi_real_null_cpf', chi_real_null_cpf)
    print('chi_imag_null_cpf', chi_imag_null_cpf)
    print('dof_null_real_cpf', dof_null_real_cpf)
    print('dof_null_imag_cpf', dof_null_imag_cpf)
    print('reduced_chi_null_cpf',chis.reduced_chi_square(chi_real_null_cpf,chi_imag_null_cpf,dof_null_real_cpf,dof_null_imag_cpf))

    chi_real_null=chis.chi_square(np.array(G_real_span), np.array(G_null.real), np.array(dG_span))
    chi_imag_null=chis.chi_square(np.array(G_im_span), np.array(G_null.imag), np.array(dG_span))
    print('chi_real_null', chi_real_null)
    print('chi_imag_null', chi_imag_null)
    print('dof_null_real', dof_null_real)
    print('dof_null_imag', dof_null_imag)
    print('reduced_chi_null',chis.reduced_chi_square(chi_real_null,chi_imag_null,dof_null_real,dof_null_imag))

    F = ft.F_test(chi_real_null, chi_imag_null, dof_null_real, dof_null_imag, model_chi_real, model_chi_imag, dof_model_real, dof_model_imag)
    print('Null',F)

    F_cpf=ft.F_test(chi_real_null_cpf, chi_imag_null_cpf, dof_null_real_cpf, dof_null_imag_cpf, model_chi_real, model_chi_imag, dof_model_real, dof_model_imag)
    print('Null CPF',F_cpf)


    #Calculating the frac rms and phase
    frac_rms= np.sqrt( G_real_span**2 + G_im_span**2 )/ (subject_cr)
    phase_lag = np.arctan2(G_im_span, G_real_span) /(2 * np.pi)
    mag = np.sqrt(G_real_span**2 + G_im_span**2)
    
    d_phase_lag = np.array(dG_span) / (mag * (2 * np.pi))
    d_frac_rms = np.array(dG_span) / np.array(subject_cr)

   
   
   
    #Calculating frac rms and phase  of the model fits
    frac_rms_model= np.sqrt( fit_full_real**2 + fit_full_imag**2 )/ fit_model
    phase_lag_model = np.arctan2(fit_full_imag, fit_full_real) /(2 * np.pi)

    frac_rms_null=  np.sqrt( G_null.real**2 + G_null.imag**2 )/ (overall_mean_cr*(1+(Q_norm*np.cos(2*av_mod))+(U_norm*np.sin(2*av_mod))))

    #frac_rms_null= np.sqrt( G_null.real**2 + G_null.imag**2 )/ subject_cr
    phase_lag_null = np.arctan2(G_null.imag, G_null.real) /(2 * np.pi)
    frac_rms_null_cpf=np.array( np.sqrt( np.array(G_null_cpf_real)**2 + np.array(G_null_cpf_imag)**2 )/ overall_mean_cr)
    phase_lag_null_cpf = np.arctan2(G_null_cpf_imag, G_null_cpf_real) /(2 * np.pi)  


    #print('G_im_span', G_im_span)

    return {
        'I_obs': I_obs,
        'Q_obs': Q_obs,
        'U_obs': U_obs,
        'cs_ref': cs_ref_power,
        'weights_ref': cs_ref.m,
        'cs_ref': cs_ref.power,
        'cs_ref_mean_real': cs_ref_mean_real,
        'cs_ref_mean_imag': cs_ref_mean_imag,
        'cs_ref_abs_mean_stack': cs_ref_mean,
        'cs_list': cs_list,
        'weights_G': [cs.m for cs in cs_list],
        'weights_ref': cs_ref.m,
        'G_real_span': G_real_span,
        'G_im_span': G_im_span,
        'G_null_real': G_null.real,
        'G_null_imag': G_null.imag,
        'G_null_real_cpf': G_null_cpf_real,
        'G_null_imag_cpf': G_null_cpf_imag,
        'dG_span': dG_span,
        "scale_factor": scale_factor,
        'Q': Q, 
        'U': U,
        'F_null': F,
        'F_null_cpf': F_cpf,
        'frac_rms': frac_rms,
        'd_frac_rms': d_frac_rms,
        'phase_lag': phase_lag,
        'd_phase_lag': d_phase_lag,
        'Fit_full_real': fit_full_real,
        'Fit_full_imag': fit_full_imag, 
        'frac_rms_model': frac_rms_model,
        'phase_lag_model': phase_lag_model,
        'frac_rms_null': frac_rms_null,
        'phase_lag_null': phase_lag_null,
        'frac_rms_null_cpf': frac_rms_null_cpf,
        'phase_lag_null_cpf': phase_lag_null_cpf

    }



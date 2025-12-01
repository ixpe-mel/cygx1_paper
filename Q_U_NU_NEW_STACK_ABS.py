
#Importing basics
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import scipy
from scipy.optimize import curve_fit

#Importing my functions
import sys
sys.path.append('/home/c2032014/py_files')
import load_and_clean as lac
import fit_rms_phase as frp
import F_test as ft
import chi_square as chis
import get_obs_file_triplets as g
import get_obs_file_triplets as gft
import G_span_abs as gs
#Importing parallel processing packages
from joblib import Parallel, delayed
import dG_span_new_abs as dgs
import chi_square as cs
import importlib
importlib.reload(ft)
#Importing stingray packages
from stingray import Lightcurve, Powerspectrum, AveragedCrossspectrum






# Example function to process one observation triplet
def process_obs_triplet(obs, Pmin, Pmax, bin_length,
                        seg_length, fmin, fmax,spur_sub,
                        norm,mod_bin_number,mod_min,mod_max,
                        coherence_corrector=None):
    


    #print(obs)
    data1, *_ = lac.load_and_clean(obs[0], Pmin, Pmax)
    data2, *_ = lac.load_and_clean(obs[1], Pmin, Pmax)
    GTI = list(np.loadtxt(obs[2]))

    I_obs = len(data1['TIME'])
    Q_obs = data1['Q']
    U_obs = data1['U']

    Q_norm = np.sum(Q_obs) / I_obs
    U_norm = np.sum(U_obs) / I_obs

    #Calculating the scale factor from deadtime nuances
    scale_factor=(I_obs+len(data2['TIME']))**2/(I_obs* len(data2['TIME']))
    #print('scale factor', scale_factor) 



    lc1 = Lightcurve.make_lightcurve(data1['TIME'], dt=bin_length, gti=GTI)
    lc2 = Lightcurve.make_lightcurve(data2['TIME'], dt=bin_length, gti=GTI)
    lc1.apply_gtis()
    lc2.apply_gtis()

    # cs ref needed for null calculation
    cs_ref = AveragedCrossspectrum.from_lightcurve(lc1, lc2, seg_length, norm='abs')
    ps_2_ref = Powerspectrum.from_lightcurve(lc2, seg_length, norm='abs')

    #plt.figure()
    #plt.plot(cs_ref.freq, cs_ref.power.real, label='Real part')
  
    aspace = np.linspace(mod_min, mod_max, mod_bin_number + 1)
    mod_min_array = aspace[:-1]
    mod_max_array = aspace[1:]
    av_mod = (mod_min_array + mod_max_array) / 2
    av_mod_err = (mod_max_array - mod_min_array) / 2
   
    G_real_span, G_im_span, n_span, m_span,lc_1_sub_span,lc_spur,cs,spur_sub_norm = gs.G_span(mod_bin_number, data1, lc2, GTI, bin_length, seg_length, fmin, fmax, spur_sub, norm)

    ps_2 = Powerspectrum.from_lightcurve(lc2, seg_length, norm='abs')
    ps_2_ref_mean =scale_factor* ps_2.power[(fmin <= ps_2.freq) & (ps_2.freq <= fmax)].mean()

    cs_ref = AveragedCrossspectrum.from_lightcurve(lc1, lc2, seg_length, norm='abs')
    cs_ref_mean_real =scale_factor* cs_ref.power.real[(fmin <= cs_ref.freq) & (cs_ref.freq <= fmax)].mean()
    cs_ref_mean_imag =scale_factor * cs_ref.power.imag[(fmin <= cs_ref.freq) & (cs_ref.freq <= fmax)].mean()
    cs_ref_mean = np.abs(cs_ref_mean_real + 1j * cs_ref_mean_imag)
    #print('cs_ref_mean_real', cs_ref_mean_real)
    
    
    
    
    
    
    G_real_span=np.array(G_real_span)*scale_factor
    G_im_span=np.array(G_im_span)*scale_factor
    
     #print('new obs cs 0', cs[0])
    #print('weights', cs[0].m)
    print('G_real_span', G_real_span)
    print('G_im_span', G_im_span)
    print('n_span', n_span)
    print('m_span', m_span)
    print('lc_1_sub_span', lc_1_sub_span)
    print('ps_2_ref_mean', ps_2_ref_mean)
    print('cs_ref_mean', cs_ref_mean)
    print('cs_ref_mean_real', cs_ref_mean_real)
    print('cs_ref_mean_imag', cs_ref_mean_imag)
    print('scale factor', scale_factor)

    
    dG = dgs.dG_span(
        G_real_span, G_im_span, lc_1_sub_span, n_span, m_span, fmin, fmax,
        seg_length, ps_2_ref_mean, cs_ref_mean_real, coherence_corrector,scale_factor
    )    
    print('dG', dG)
   

    return {
        "obs_name": obs,
        "I_obs": I_obs,
        "Q_obs": Q_obs,
        "U_obs": U_obs,
        "Q_norm": Q_norm,
        "U_norm": U_norm,
        "cs_ref": cs_ref,
        "cs_G": cs,
        "weights_ref" : cs_ref.m,
        "weights_G": [cs.m for cs in cs],
        "ps_2_ref": ps_2_ref,
        "scale factor": scale_factor,
        "dG": dG,
        "lc_1_sub_span": lc_1_sub_span,
        "lc_ref": lc2

    }
#from each obs triplet we get the lc1subspan ie 20 lightcurves for each mod angle.
#We want to take stack the lightcurves for each mod angle bin to treat them as one observation
#We can do this by averaging the lightcurves in each bin weighted by their count rate

#Define sinusoidal models to fit Re[G] and Im[G] 


#Real part of FUll model
def cross_spec_model_real(phi,B,C,A,J):
    Re_G=(1/J) * ( A + (B*np.cos(2*phi)) + (C*np.sin(2*phi)) )
    return Re_G

#Im part of full model
# The imaginary sinusoid does not have the A term
def cross_spec_model_imag(phi,B,C,J):
    J=20
    Im_G=(1/J) * ( (B*np.cos(2*phi)) + (C*np.sin(2*phi)) )
    return Im_G




# Null hypothesis model: No polarisation variability
def cross_spec_model_null(phi,C_nu_mag_sqrd,Q_norm,U_norm,J):#, C_nu_mag_sqrd,Q_norm,U_norm,J):
 
    return (1/J) * C_nu_mag_sqrd * (1 + Q_norm * np.cos(2 * phi) + U_norm * np.sin(2 * phi))



# 'Null null' hypothesis model: Constant polarised flux
def cross_spec_model_null_cpf(phi,J):
    J=20
    C_nu_mag_sqrd= cs_ref_abs_mean_stack
    return (1/J) * C_nu_mag_sqrd


def sum_stingray_lightcurves(lightcurves, dt=None):
    """
    Sum multiple Stingray Lightcurve objects.
    
    Parameters
    ----------
    lightcurves : list of Lightcurve
        List of Stingray Lightcurve objects to sum.
    dt : float, optional
        Time resolution for resampling (if needed). If None, uses the smallest dt among all lightcurves.
    
    Returns
    -------
    summed_lc : Lightcurve
        Summed Stingray Lightcurve object.
    """
    # Determine common dt
    if dt is None:
        dt = min(lc.dt for lc in lightcurves)
    
    # Determine common time range
    t_start = min(lc.tstart for lc in lightcurves)
    t_end = max(lc.tend for lc in lightcurves)
    
    # Create a common time array
    common_time = np.arange(t_start, t_end, dt)
    
    # Interpolate each light curve onto the common time grid
    interpolated_counts = []
    for lc in lightcurves:
        interp = np.interp(common_time, lc.time, lc.counts, left=0, right=0)
        interpolated_counts.append(interp)
    
    # Sum the counts
    summed_counts = np.sum(interpolated_counts, axis=0)
    
    # Create summed Stingray Lightcurve
    summed_lc = Lightcurve(common_time, summed_counts, dt=dt)
    
    return summed_lc




def run_q_u_nu_stack(obs_folder, obs_names, Pmin, Pmax, bin_length,
                                seg_length, fmin, fmax,f_bin_number,
                                mod_bin_number,mod_min,mod_max, spur_sub, coherence_corrector,
                                norm, output_file):
    obs_triplets = gft.get_obs_file_pairs(obs_folder, obs_names)
    # Parallel processing of obs_triplets
    
    
    #Getting cross spectral results from all observations
    results = Parallel(n_jobs=-1)(delayed(process_obs_triplet)(
            obs, Pmin, Pmax, bin_length, seg_length, fmin, fmax,
            spur_sub, norm, mod_bin_number,mod_min,mod_max,
            coherence_corrector=False
        ) for obs in obs_triplets)
   
    
    #Creating modulation bins
    aspace = np.linspace(mod_min, mod_max, mod_bin_number + 1)
    mod_min_array = aspace[:-1]
    mod_max_array = aspace[1:]
    av_mod = (mod_min_array + mod_max_array) / 2
    av_mod_err = (mod_max_array - mod_min_array) / 2

    #Making frequency list
    fspace = np.linspace(fmin, fmax, f_bin_number + 1)
    f_min_array, f_max_array = fspace[:-1], fspace[1:]
    f_angle_list = list(zip(f_min_array, f_max_array)) #Equally spaced frequency bins
    av_f = (f_min_array + f_max_array) / 2
    av_f_err = (f_max_array - f_min_array) / 2
   

    # Unpack the results from each observation
    dG_arr= np.array([r["dG"] for r in results])
    I_obs_arr = [r["I_obs"] for r in results]
    Q_obs_arr = [r["Q_obs"] for r in results]
    U_obs_arr = [r["U_obs"] for r in results]
    cs_ref_obs = [r["cs_ref"] for r in results]  
    cs_G_obs = [r["cs_G"] for r in results]
    weights_ref_obs = [r["weights_ref"] for r in results]
    weights_G_obs = [r["weights_G"] for r in results]
    ps_2_ref = [r["ps_2_ref"] for r in results]
    scale_factors = [r["scale factor"] for r in results]
    obs_names = [r["obs_name"] for r in results]
    cs_G_obs_real = [[cs.power.real for cs in sublist] for sublist in cs_G_obs]
    cs_G_obs_imag = [[cs.power.imag for cs in sublist] for sublist in cs_G_obs]
    lc_1_sub_span = [r["lc_1_sub_span"] for r in results]  # Assuming all observations have the same lc_1_sub_span structure
    lc_ref_span=[r["lc_ref"] for r in results]




# Assume lc_1_sub_span is a list of lists of LC objects with .counts and .time attributes

    # Determine a common length (you could also choose a fixed number if you prefer)
    target_len = max(lc.counts.shape[0] for obs in lc_1_sub_span for lc in obs)

    summed_lc1 = []
    for mod_angle_idx in range(mod_bin_number):
        interp_lcs = []

        for obs_idx in range(len(lc_1_sub_span)):
            lc_obj = lc_1_sub_span[obs_idx][mod_angle_idx]

            # Original time grid normalized 0-1
            x = np.linspace(0, 1, len(lc_obj.counts))
            # Target grid
            x_target = np.linspace(0, 1, target_len)

            # Interpolate counts
            interp_counts = np.interp(x_target, x, lc_obj.counts)
            interp_lcs.append(interp_counts)

        # Sum interpolated counts across observations
        summed_counts = np.sum(interp_lcs, axis=0)

        # Create a new LC object
        # Keep the time axis of the first observation (rescaled to target length)
        # Assuming your LC object can be initialized like LC(time, counts)
        summed_time = np.linspace(lc_1_sub_span[0][mod_angle_idx].time[0],
                                lc_1_sub_span[0][mod_angle_idx].time[-1],
                                target_len)

        summed_lc = lc_1_sub_span[0][mod_angle_idx].__class__(summed_time, summed_counts)
        summed_lc1.append(summed_lc)


        # lc_list: a list of light curve objects, each with .time and .counts
        target_len_lc2 = max(lc2.counts.shape[0] for lc2 in lc_ref_span)

        # Interpolate all LCs onto the same target grid and sum
        interp_lcs_2 = []

        for lc in lc_ref_span:
            # Original time grid normalized 0-1
            x = np.linspace(0, 1, len(lc.counts))
            # Target grid
            x_target = np.linspace(0, 1, target_len_lc2)
            
            # Interpolate counts
            interp_counts = np.interp(x_target, x, lc.counts)
            interp_lcs_2.append(interp_counts)
        # Sum interpolated counts across all LCs
        summed_counts_2 = np.sum(interp_lcs_2, axis=0)

        # Create a new LC object
        # Use the time axis of the first LC rescaled to target length
        summed_time_2 = np.linspace(lc_ref_span[0].time[0],
                                lc_ref_span[0].time[-1],
                                target_len_lc2)

        # Assuming your LC class can be initialized like LC(time, counts)
        summed_lc_2 = lc_ref_span[0].__class__(summed_time_2, summed_counts_2)
        lc2_meanrate=summed_lc_2.meanrate




    #summed_lc_2 = sum_stingray_lightcurves([lc_ref_span[obs_idx] for obs_idx in range(len(lc_ref_span))])
    #lc2_meanrate=summed_lc_2.meanrate
#Now to sum the reference band lightcurves



    #summing lc1 sub spans over observations for each mod angle
    #summed_lc1 = []
    #for mod_angle_idx in range(mod_bin_number):
    #    summed_lc = sum(lc_1_sub_span[obs_idx][mod_angle_idx].counts for obs_idx in range(len(lc_1_sub_span)))
    #    summed_lc1.append(summed_lc)


    # Applying scale factors to cross-spectral data
    cs_G_obs_real=np.array(cs_G_obs_real)*np.array( scale_factors)[:, None, None]
    cs_G_obs_imag=np.array(cs_G_obs_imag)*np.array( scale_factors)[:, None, None]

   
    # Unpacking real and imaginary parts of reference cross-spectrum and applying scale factors
    cs_ref_obs_real = [cs_ref.power.real for cs_ref in cs_ref_obs]
    cs_ref_obs_real = np.array(cs_ref_obs_real) * np.array(scale_factors)[:, None]
    cs_ref_obs_imag = [cs_ref.power.imag for cs_ref in cs_ref_obs]
    cs_ref_obs_imag = np.array(cs_ref_obs_imag) * np.array(scale_factors)[:, None]
    cs_ref_freqs = [cs_ref.freq for cs_ref in cs_ref_obs]

    
    #Calculating normalised Stokes parameters over all observations
    I_tot=np.sum(I_obs_arr)
    Q_norm=sum(sum(sublist) for sublist in Q_obs_arr)/I_tot
    U_norm=sum(sum(sublist) for sublist in U_obs_arr)/I_tot


    #Defining weights for stacking
    weights_expanded_obs = np.array(weights_G_obs)[:, :, np.newaxis]
    numerator_obs_real = np.sum(cs_G_obs_real * weights_expanded_obs, axis=0)       
    denominator_obs_real = np.sum(weights_expanded_obs, axis=0)                    
    numerator_obs_imag = np.sum(cs_G_obs_imag * weights_expanded_obs, axis=0)       
    denominator_obs_imag = np.sum(weights_expanded_obs, axis=0)   

    # Final weighted average
    cs_G_real_stacked = numerator_obs_real / np.where(denominator_obs_real == 0, np.nan, denominator_obs_real)
    cs_G_imag_stacked = numerator_obs_imag / np.where(denominator_obs_imag == 0, np.nan, denominator_obs_imag)
    cs_ref_real_stacked=np.average(cs_ref_obs_real,weights=weights_ref_obs,axis=0)
    cs_ref_imag_stacked=np.average(cs_ref_obs_imag,weights=weights_ref_obs,axis=0)


    #subject_cr=[i.meanrate for i in lc_1_sub_span]

    #Propagating errors in dG
  
    dG_arr=np.sqrt( np.sum( (np.array(weights_G_obs)**2 * dG_arr**2),axis=0 )/np.sum(np.array(weights_G_obs),axis=0)**2 )



    results_freq = Parallel(n_jobs=-1)(delayed(process_frequency_bin)(i,cs_G_real_stacked,cs_G_imag_stacked,cs_ref_real_stacked,cs_ref_imag_stacked,cs_G_obs[0][0].freq,av_mod,Q_norm,
                            U_norm,dG_arr,summed_lc1,lc2_meanrate) for i in (f_angle_list))
    
    np.save(output_file, np.array(results_freq))

    return np.array(results_freq), av_mod, av_mod_err, av_f, av_f_err, Q_norm, U_norm, cs_G_real_stacked, cs_G_imag_stacked, cs_ref_real_stacked, cs_ref_imag_stacked, cs_ref_freqs



# Now to calculate Q,U in each frequency bin
def process_frequency_bin(i,cs_G_real_stacked,cs_G_imag_stacked,cs_ref_real_stacked,cs_ref_imag_stacked,cs_freqs,av_mod,Q_norm,U_norm,dG,summed_lc1,lc_2_meanrate):

    #Averaging G_real and G_imag over the frequency bin

 

    G_real=[cs_G_real_stacked[(i[0]<=cs_freqs) & (cs_freqs<=i[1])].mean() for cs_G_real_stacked in cs_G_real_stacked]
    G_imag=[cs_G_imag_stacked[(i[0]<=cs_freqs) & (cs_freqs<=i[1])].mean() for cs_G_imag_stacked in cs_G_imag_stacked]
    #Averaging reference cs over the frequency bin
    cs_ref_real_mean=np.array(cs_ref_real_stacked[(i[0] <= cs_freqs) & (cs_freqs <= i[1])].mean())
    cs_ref_imag_mean=np.array(cs_ref_imag_stacked[(i[0] <= cs_freqs) & (cs_freqs <= i[1])].mean())

  
    cs_ref_average=cs_ref_real_mean+1j*cs_ref_imag_mean
    cs_ref_abs_mean_stack=np.abs(cs_ref_average)


    J=20 #Number of modulation angle bins 
    
    def cross_spec_model_real_fixedJ(phi,B,C):
        return cross_spec_model_real(phi, B, C, A=cs_ref_abs_mean_stack, J=J)
    
    def cross_spec_model_imag_fixedJ(phi, B, C):
        return cross_spec_model_imag(phi, B, C, J=J)
    
    def cross_spec_model_null_fixedJ(phi, C_nu_mag_sqrd, Q_norm, U_norm):
        return cross_spec_model_null(phi, C_nu_mag_sqrd, Q_norm, U_norm, J=J)
    

    #Ftting full model to real and imaginary parts of G
    parameters_real,pcovreal=curve_fit(cross_spec_model_real_fixedJ,np.array(av_mod),np.array(G_real),sigma=dG)
    parameters_imag,pcovimag=curve_fit(cross_spec_model_imag_fixedJ,np.array(av_mod),np.array(G_imag),sigma=dG)

    parameters_real,pcovreal=curve_fit(cross_spec_model_real_fixedJ,np.array(av_mod),np.array(G_real),sigma=dG)
    parameters_imag,pcovimag=curve_fit(cross_spec_model_imag_fixedJ,np.array(av_mod),np.array(G_imag),sigma=dG)
    print('parameters_real', parameters_real)
    print('parameters_imag', parameters_imag)


    fit_y_model_real = cross_spec_model_real_fixedJ(av_mod, parameters_real[0], parameters_real[1])
    fit_y_model_imag=cross_spec_model_imag_fixedJ(av_mod,parameters_imag[0],parameters_imag[1])

    dof_model_real=len(av_mod)-2 #Free params are B and C (Q and U)
    dof_model_imag=len(av_mod)-2


    model_chi_real=cs.chi_square(G_real,fit_y_model_real,dG)
    print('model_chi_real', model_chi_real)
    print('dof_model_real', dof_model_real)


    model_chi_imag=cs.chi_square(G_imag,fit_y_model_imag,dG)
    print('model_chi_imag', model_chi_imag)
    print('dof_model_imag', dof_model_imag)
    
    print(r'$\chi^2$ of Full polarisation variability model',model_chi_real+model_chi_imag)
    print('reduced chi sqr full model', (model_chi_real+model_chi_imag)/(dof_model_real+dof_model_imag) )
    #Unpacking full model parameters
    B_real, C_real = parameters_real
    B_imag, C_imag = parameters_imag

    Breal_err,Creal_err= np.sqrt(np.diag(pcovreal))
    Bimag_err,Cimag_err= np.sqrt(np.diag(pcovimag))
      
    #Calculating null hypothesis
    G_null=cross_spec_model_null_fixedJ(np.array(av_mod),cs_ref_abs_mean_stack,Q_norm,U_norm)
    G_null_real=G_null.real
    G_null_imag=G_null.imag
    G_null_real_dof= len(av_mod)
    G_null_imag_dof= len(av_mod)
    chi_null_real= chis.chi_square(G_real,G_null_real , dG)
    chi_null_imag= chis.chi_square(G_imag,G_null_imag , dG)

    print(r'$\chi^2$ of Null Hypothesis',chi_null_real+chi_null_imag)

    #Calculating the null hypothesis of the constant polarised flux
    G_null_cpf_real=cs_ref_abs_mean_stack/J
    G_null_cpf_imag=0
    G_null_cpf_real_dof= len(av_mod)
    G_null_cpf_imag_dof= len(av_mod)
    chi_cpf_real= chis.chi_square(G_real, G_null_cpf_real,dG )
    chi_cpf_imag= chis.chi_square(G_imag, G_null_cpf_imag,dG )

    print(r'$\chi^2$ of Constant Polarised Flux',chi_cpf_real+chi_cpf_imag)

    #Performing F-tests
    F=ft.F_test(chi_null_real,chi_null_imag,G_null_real_dof,G_null_imag_dof,
                model_chi_real,model_chi_imag,dof_model_real,dof_model_imag)
    print('F test: Full Model vs Null',F)

    F_null_null=ft.F_test(chi_cpf_real,chi_cpf_imag,G_null_cpf_real_dof,G_null_cpf_imag_dof,
                         model_chi_real,model_chi_imag,dof_model_real,dof_model_imag)

    print('F test: Full model vs  Null Hypothesis CPF',F_null_null)



    #Null hypothesis co-efficients (plotting purposes)
    A_null_real=cs_ref_abs_mean_stack #/J
    B_null_real=cs_ref_abs_mean_stack*Q_norm 
    C_null_real=cs_ref_abs_mean_stack*U_norm
    B_null_imag=0
    C_null_imag=0

    #Null null hypothesis coefficients
    def sinusoid(phi, A, B, C):
        return A + (B * np.cos(2 * phi)) + (C * np.sin(2 * phi))

 
    f_av = (i[0] + i[1]) / 2
    subject_cr=[i.meanrate for i in summed_lc1]
    overall_mean_cr=np.mean(subject_cr)


    popt, pcov = curve_fit(sinusoid, av_mod, subject_cr)
    A, B, C = popt

    fit_model= sinusoid(av_mod, A, B, C)

    G_real=np.array(G_real)
    G_im=np.array(G_imag)


        #Calculating the frac rms and phase
    frac_rms= np.sqrt( G_real**2 + G_im**2 )/ (subject_cr)
    phase_lag = np.arctan2(G_im, G_real) /(2 * np.pi)
    mag = np.sqrt(G_real**2 + G_im**2)
    d_phase_lag = np.array(dG) / (mag * (2 * np.pi))
    d_frac_rms = np.array(dG) / np.array(subject_cr)


    #Calculating frac rms and phase  of the model fits
    frac_rms_model= np.sqrt( fit_y_model_real**2 + fit_y_model_imag**2 )/ fit_model
    phase_lag_model = np.arctan2(fit_y_model_imag, fit_y_model_real) /(2 * np.pi)

    frac_rms_null=  np.sqrt( G_null.real**2 + G_null.imag**2 )/ (overall_mean_cr*(1+(Q_norm*np.cos(2*av_mod))+(U_norm*np.sin(2*av_mod))))

    #frac_rms_null= np.sqrt( G_null.real**2 + G_null.imag**2 )/ subject_cr
    phase_lag_null = np.arctan2(G_null.imag, G_null.real) /(2 * np.pi)
    frac_rms_null_cpf=np.array( np.sqrt( np.array(G_null_cpf_real)**2 + np.array(G_null_cpf_imag)**2 )/ overall_mean_cr)
    phase_lag_null_cpf = np.arctan2(G_null_cpf_imag, G_null_cpf_real) /(2 * np.pi)  

    #Phase of GQ and GU
    phase_GQ=np.arctan2(B_imag,B_real)/(2*np.pi)
    phase_GU=np.arctan2(C_imag,C_real)/(2*np.pi)

    phase_GQ_err= np.sqrt( (B_real*Bimag_err)**2 + (B_imag*Breal_err)**2 )/((B_real**2 + B_imag**2)*2*np.pi)
    phase_GU_err= np.sqrt( (C_real*Cimag_err)**2 + ( C_imag*Creal_err)**2 )/((C_real**2 + C_imag**2)*2*np.pi)

    frac_rms_GQ= np.sqrt( B_real**2 + B_imag**2 )/ lc_2_meanrate*np.sqrt(cs_ref_real_mean)
    frac_rms_GU= np.sqrt( C_real**2 + C_imag**2 )/ lc_2_meanrate*np.sqrt(cs_ref_real_mean)
    
    frac_rms_GQ_err= np.sqrt( (B_real*Breal_err)**2 + (B_imag*Bimag_err)**2 )/( np.sqrt( B_real**2 + B_imag**2 )* lc_2_meanrate* np.sqrt(cs_ref_real_mean) )
    frac_rms_GU_err= np.sqrt( (C_real*Creal_err)**2 + (C_imag*Cimag_err)**2 )/( np.sqrt( C_real**2 + C_imag**2 )* lc_2_meanrate* np.sqrt(cs_ref_real_mean) )
    
    phase_GQ_null=np.arctan2(B_null_imag,B_null_real)/(2*np.pi)
    phase_GU_null=np.arctan2(C_null_imag,C_null_real)/(2*np.pi)

    frac_rms_GQ_null=np.sqrt( B_null_real**2 + B_null_imag**2 )/ lc_2_meanrate*np.sqrt(cs_ref_real_mean)
    frac_rms_GU_null=np.sqrt( C_null_real**2 + C_null_imag**2 )/ lc_2_meanrate*np.sqrt(cs_ref_real_mean)

    
   

    #Packaging results into a dictionary
    result = {
    "f_av": f_av,
    "B_real_err": Breal_err,
    "C_real_err": Creal_err,
    "B_imag_err": Bimag_err,
    "C_imag_err": Cimag_err,
    "B_real": B_real,
    "C_real": C_real,
    "B_imag": parameters_imag[0],
    "C_imag": parameters_imag[1],
    "A_null_real": A_null_real,
    "B_null_real": B_null_real,
    "C_null_real": C_null_real,
    "B_null_imag": B_null_imag,
    "C_null_imag": C_null_imag,
    "cs_G_real_stacked": cs_G_real_stacked,
    "cs_G_imag_stacked": cs_G_imag_stacked,
    "cs_ref_real_stacked": cs_ref_real_stacked,
    "cs_ref_imag_stacked": cs_ref_imag_stacked,
    "F":F,
    "F_null_null": F_null_null,
    "G_real": G_real,
    "G_imag": G_imag,
    "G_null_real": G_null_real,
    "G_null_imag": G_null_imag,
    "G_null_cpf_real": G_null_cpf_real,
    "G_null_cpf_imag": G_null_cpf_imag,
    "fit_model_real": fit_y_model_real,
    "fit_model_imag": fit_y_model_imag,
    "dG": dG,
    "Q_norm": Q_norm,
    "U_norm": U_norm,
    "cs_ref_abs_mean_stack": cs_ref_abs_mean_stack,
    "frac_rms_model": frac_rms_model,
    "phase_lag_model": phase_lag_model,
    "frac_rms_null": frac_rms_null,
    "phase_lag_null": phase_lag_null,
    "frac_rms_null_cpf": frac_rms_null_cpf,
    "phase_lag_null_cpf": phase_lag_null_cpf,
    "frac_rms": frac_rms,
    "phase_lag": phase_lag,
    "d_frac_rms": d_frac_rms,
    "d_phase_lag": d_phase_lag,
    "phase_GQ": phase_GQ,   
    "phase_GU": phase_GU,
    "frac_rms_GQ": frac_rms_GQ,
    "frac_rms_GU": frac_rms_GU,
    "phase_GQ_null": phase_GQ_null,
    "phase_GU_null": phase_GU_null,
    "frac_rms_GQ_null": frac_rms_GQ_null,
    "frac_rms_GU_null": frac_rms_GU_null,
    "phase_GQ_err": phase_GQ_err,
    "phase_GU_err": phase_GU_err,
    "frac_rms_GQ_err": frac_rms_GQ_err,
    "frac_rms_GU_err": frac_rms_GU_err


    }   

    return result

from joblib import Parallel, delayed
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
from joblib import Parallel, delayed
from scipy.optimize import curve_fit
import chi_square as cs
importlib.reload(ft)


def cross_spec_model_real(phi,A,B,C,J):
    Re_G=(1/J) * ( A + (B*np.cos(2*phi)) + (C*np.sin(2*phi)) )
    return Re_G

#The imaginary sinusoid does not have the A term
def cross_spec_model_imag(phi,B,C,J):
    Im_G=(1/J) * ( (B*np.cos(2*phi)) + (C*np.sin(2*phi)) )
    return Im_G




def cross_spec_model_null(phi,Q_norm,U_norm ,C_nu_mag_sqrd,J):
    return (1/J) * C_nu_mag_sqrd * (1 + Q_norm * np.cos(2 * phi) + U_norm * np.sin(2 * phi))


#def cross_spec_model_null_fixedJ(phi, Q_norm, U_norm ,C_nu_mag_sqrd):
#        return cross_spec_model_null(phi, C_nu_mag_sqrd, Q_norm, U_norm, J=J)



def cross_spec_model_null_cpf(J,C_nu_mag_sqrd):
    return C_nu_mag_sqrd / J





def process_obs(obs, Pmin, Pmax, bin_length, seg_length, fmin, fmax, spur_sub, norm,
                 mod_bin_number, coherence_corrector, plot=True):
    
   

    aspace = np.linspace(np.radians(-90), np.radians(90), mod_bin_number + 1)
    mod_min_array = aspace[:-1]
    mod_max_array = aspace[1:]
    av_mod = (mod_min_array + mod_max_array) / 2
    av_mod_err = (mod_max_array - mod_min_array) / 2
    J=mod_bin_number


    data1, *_ = lac.load_and_clean(obs[0], Pmin, Pmax)
    data2, *_ = lac.load_and_clean(obs[1], Pmin, Pmax)
    GTI = list(np.loadtxt(obs[2]))

    I_obs = len(data1['TIME'])
    Q_obs = data1['Q']
    U_obs = data1['U']
    
    scale_factor=(I_obs+len(data2['TIME']))**2/(I_obs* len(data2['TIME']))
    print('scale_factor',scale_factor)

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

    G_real_span = np.array(G_real_span) * scale_factor
    G_im_span = np.array(G_im_span) * scale_factor

    Q_norm = np.sum(Q_obs) / I_obs
    U_norm = np.sum(U_obs) / I_obs
    Q=np.sum(Q_obs)
    U=np.sum(U_obs)

    #G_null = cross_spec_model_null(J,Q_uncorrected,U_uncorrected,np.array(av_mod), cs_ref_mean)
    G_null= cross_spec_model_null(J,Q_norm,U_norm,np.array(av_mod), cs_ref_mean)


    G_null_cpf_real =cs_ref_mean_real/mod_bin_number
    G_null_cpf_real=[G_null_cpf_real]*len(av_mod)
    
    G_null_cpf_imag=0
    G_null_cpf_imag=np.array([G_null_cpf_imag]*len(av_mod))
    #G_null_cpf_imag=G_null_cpf_imag[1:])
    #print('G_null_cpf',G_null_cpf)
    
    print('G_real_span', G_real_span)
    print('G_im_span', G_im_span)
    print('n_span', n_span)
    print('m_span', m_span)
    print('lc_1_sub_span', lc_1_sub_span)
    print('ps_2_ref_mean', ps_2_ref_mean)
    print('cs_ref_mean', cs_ref_mean)
    print('cs_ref_mean_real', cs_ref_mean_real)

    dG_span = dgs.dG_span(
        G_real_span, G_im_span, lc_1_sub_span, n_span, m_span, fmin, fmax,
        seg_length, ps_2_ref_mean, cs_ref_mean_real, coherence_corrector
    )

    # Performing ftest on each observation

    def cross_spec_model_real(phi,B,C):
        global cs_ref_mean_real
        global J
        Re_G=(1/J) * ( cs_ref_mean + (B*np.cos(2*phi)) + (C*np.sin(2*phi)) )
        return Re_G

    def cross_spec_model_imag(phi,B,C):
        global J
        Im_G=(1/J) * ( (B*np.cos(2*phi)) + (C*np.sin(2*phi)) )
        return Im_G

    params_full_fit_real, pcov_full_real=curve_fit(cross_spec_model_real,np.array(av_mod), np.array(G_real_span), np.array(dG_span))
    params_full_fit_imag, pcov_full_imag=curve_fit(cross_spec_model_imag,np.array(av_mod), np.array(G_im_span), np.array(dG_span))

    fit_full_real = cross_spec_model_real(av_mod, params_full_fit_real[0], params_full_fit_real[1])
    fit_full_imag = cross_spec_model_imag(av_mod, params_full_fit_imag[0], params_full_fit_imag[1])

    dof_model_real=len(av_mod)-2 #Free params are B and C (Q and U)
    dof_model_imag=len(av_mod)-2

    model_chi_real=cs.chi_square(G_real,fit_full_real,dG)
    print('model_chi_real', model_chi_real)
    print('dof_model_real', dof_model_real)

    model_chi_imag=cs.chi_square(G_im,fit_full_imag,dG)
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

    chi_real_null=chis.chi_square(np.array(G_real_span), np.array(G_null.real), np.array(dG_span))
    chi_imag_null=chis.chi_square(np.array(G_im_span), np.array(G_null.imag), np.array(dG_span))

    F = ft.F_test(chi_real_null, chi_imag_null, dof_null_real, dof_null_imag, model_chi_real, model_chi_imag, dof_model_real, dof_model_imag)
    print(F)

    F_cpf=ft.F_test(chi_real_null_cpf, chi_imag_null_cpf, dof_null_real_cpf, dof_null_imag_cpf, model_chi_real, model_chi_imag, dof_model_real, dof_model_imag)
    #print(F_cpf)

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
        'fit_sin180_real': fit_sin180_real,
        'fit_sin180_imag': fit_sin180_imag,
        "scale_factor": scale_factor,
        'Q': Q, 
        'U': U,
        
    }




def run_QPO_finder_absolute_stacked(obs_folder, obs_names, Pmin, Pmax, bin_length,
                                seg_length, fmin, fmax,
                                mod_bin_number, spur_sub, coherence_corrector,
                                output_file):
    """
    Main function to run the QPO finder for absolute normalisation.
    
    Args:
        obs_folder (str): Path to the folder containing observations.
        obs_names (list): List of observation directory names.
        Pmin (float): Minimum p channel.
        Pmax (float): Maximum p channel.
        bin_length (float): Length of bins used in lcs.
        seg_length (float): Length of segments used for averaging cs.
        fmin (float): Minimum frequency of variability.
        fmax (float): Maximum frequency of variability.
        mod_min (float): Minimum modulation angle.
        mod_max (float): Maximum modulation angle.
        mod_bin_number (int): Number of modulation angle bins.
        spur_sub (bool): Whether to subtract spurious polarisation.
        coherence_corrector (bool): Whether to apply coherence correction.
        output_file (str): Path to save the output file.

    Returns:
        None
    """
    obs_triplets = gft.get_obs_file_pairs(obs_folder, obs_names)
    results_par = Parallel(n_jobs=-1)(delayed(process_obs)(
        obs, Pmin, Pmax, bin_length, seg_length, fmin, fmax,
        spur_sub, norm='abs', mod_bin_number=mod_bin_number,
        coherence_corrector=coherence_corrector
    ) for obs in obs_triplets)


    # Collect results
    I_obs_arr = []
    Q_obs_arr = []
    U_obs_arr = []
    cs_ref_obs = []
    weights_ref = []
    cs_ref_real_mean_arr = []
    cs_ref_im_mean_arr = []
    cs_ref_abs_mean_stack = []
    cs_G_obs = []
    weights_G_arr = []
    G_real_span_arr = []
    G_im_span_arr = []
    G_real_null_arr = []
    G_im_null_arr = []
    dG_span_arr = []
    G_null_real_cpf_arr = []
    G_null_imag_cpf_arr = []
    fit_sin180_real_arr = []
    fit_sin180_imag_arr = []
    scale_factors_arr = []
    Q_arr=[]
    U_arr=[]
    


    # Unpack results
    for res in results_par:
        I_obs_arr.append(res['I_obs'])
        Q_obs_arr.append(res['Q_obs'])
        U_obs_arr.append(res['U_obs'])
        cs_ref_obs.append(res['cs_ref'])
        weights_ref.append(res['weights_ref'])
        cs_ref_real_mean_arr.append(res['cs_ref_mean_real'])
        cs_ref_im_mean_arr.append(res['cs_ref_mean_imag'])
        cs_ref_abs_mean_stack.append(res['cs_ref_abs_mean_stack'])
        cs_G_obs.append(res['cs_list'])
        weights_G_arr.append(res['weights_G'])
        G_real_span_arr.append(res['G_real_span'])
        G_im_span_arr.append(res['G_im_span'])
        G_real_null_arr.append(res['G_null_real'])
        G_im_null_arr.append(res['G_null_imag'])
        dG_span_arr.append(res['dG_span'])
        G_null_real_cpf_arr.append(res['G_null_real_cpf'])
        G_null_imag_cpf_arr.append(res['G_null_imag_cpf'])
        fit_sin180_real_arr.append(res['fit_sin180_real'])
        fit_sin180_imag_arr.append(res['fit_sin180_imag'])
        scale_factors_arr.append(res['scale_factor'])
        Q_arr.append(res['Q'])
        U_arr.append(res['U'])



    #scale_factors_arr= np.array(scale_factors_arr)[:, np.newaxis]
    Q=np.sum(Q_arr)
    U=np.sum(U_arr)
    

    aspace = np.linspace(np.radians(-90), np.radians(90), mod_bin_number + 1)
    mod_min_array = aspace[:-1]
    mod_max_array = aspace[1:]
    av_mod_arr = (mod_min_array + mod_max_array) / 2
    av_mod_err = (mod_max_array - mod_min_array) / 2

    #Plotting individual g

 #   for i in range(len(obs_triplets)):
 #       plt.figure()
 #       plt.title('real')
 #       plt.plot(av_mod_arr,fit_sin180_real_arr[i], label='fit_sin180_real')
 #       plt.errorbar(av_mod_arr, G_real_span_arr[i], yerr=dG_span_arr[i], ls='None', label=f'Obs {i+1}')
 #       plt.plot(av_mod_arr, G_real_null_arr[i],label='G_null_real')
 #       plt.plot(av_mod_arr,G_null_real_cpf_arr[i], label='G_null_real_cpf')
 #       plt.legend()
 #       plt.show()
      
        
 #       plt.figure()
 #       plt.title('imaginary')
 #       plt.plot(av_mod_arr,fit_sin180_imag_arr[i], label='fit_sin180_imag')
 #       plt.errorbar(av_mod_arr, G_im_span_arr[i], yerr=dG_span_arr[i], ls='None', label=f'Obs {i+1}')
 #       plt.plot(av_mod_arr, G_im_null_arr[i], label='G_null_imag')
 #       plt.plot(av_mod_arr,G_null_imag_cpf_arr[i], label='G_null_imag_cpf')
 #       #plt.title(f'Observation {obs_triplets[i]}')
 #       plt.legend()
 #       plt.show()
    J=mod_bin_number
    def cross_spec_model_real_fixedJ(phi, A, B, C):
        return cross_spec_model_real(phi, A, B, C, J=J)
    
    def cross_spec_model_imag_fixedJ(phi, B, C):
        return cross_spec_model_imag(phi, B, C, J=J)
    
    def cross_spec_model_null_fixedJ(phi,  Q_norm, U_norm,C_nu_mag_sqrd):
        return cross_spec_model_null(phi, Q_norm, U_norm,C_nu_mag_sqrd, J=J)

    I_tot=np.sum(I_obs_arr)
    Q_norm=Q/I_tot
    U_norm=U/I_tot



    I_tot=np.sum(I_obs_arr)

    #Q_norm=sum(Q_obs_arr)/I_tot
   # U_norm=sum(U_obs_arr)/I_tot
    #print('weights_G_arr',weights_G_arr)
    #print('G_real_span_arr',G_real_span_arr)
    G_real=np.average(G_real_span_arr,weights=weights_G_arr,axis=0)
    print('G_real',G_real)
    G_imag=np.average(G_im_span_arr,weights=weights_G_arr,axis=0)



    cs_ref_abs_mean_stack=np.average(cs_ref_real_mean_arr,weights=weights_ref,axis=0)

    #G_null_real=np.average(G_real_null_arr,weights=weights_G_arr,axis=0)
    #G_null_imag=np.average(G_im_null_arr,weights=weights_G_arr,axis=0)


    #G_null_real_cpf=np.average(G_null_real_cpf_arr,weights=weights_G_arr,axis=0)
    #G_null_imag_cpf=np.average(G_null_imag_cpf_arr,weights=weights_G_arr,axis=0)

    dG=np.array(dG_span_arr)
    print('dG',dG)
    #Summing the errors in quadrate
    dG=np.sqrt(  np.sum(  dG**2,axis=0 )      )/len(obs_triplets)
    print('dG',dG)

    parameters_real,pcovreal=curve_fit(cross_spec_model_real_fixedJ,np.array(av_mod_arr),np.array(G_real))
    parameters_imag,pcovimag=curve_fit(cross_spec_model_imag_fixedJ,np.array(av_mod_arr),np.array(G_imag))
    
    fit_y_model_real = cross_spec_model_real_fixedJ(av_mod_arr, parameters_real[0], parameters_real[1],parameters_real[2])
    fit_y_model_imag=cross_spec_model_imag_fixedJ(av_mod_arr,parameters_imag[0],parameters_imag[1])


    #Calculating chi sqr value of 90 sine fit
    dof_model_real=len(av_mod_arr)-3
    dof_model_imag=len(av_mod_arr)-3


    model_chi_real=cs.chi_square(G_real,fit_y_model_real,dG)#phase,phase_err,fit_y_line_phase,dof_line)
    model_chi_imag=cs.chi_square(G_imag,fit_y_model_imag,dG)
    reduced_chi_model=cs.reduced_chi_square(model_chi_real,model_chi_imag,dof_model_real,dof_model_imag)
    

   
   # _,fit_sin180_real,_,fit_sin180_imag,dof_sin180_real, dof_sin180_imag, real_sin180_chi, imag_sin180_chi,*_= frp.fit_sine_180(np.array(av_mod_arr), np.array(G_real), np.array(dG), np.array(G_imag), np.array(dG))
 

    "Constant PD and PA null hypothesis"
    #print('cs_ref_abs_mean_stack',cs_ref_abs_mean_stack)
    #print('Q_norm',Q_norm)
    #print('U_norm',U_norm)

    G_null=cross_spec_model_null_fixedJ(np.array(av_mod_arr),Q_norm,U_norm,cs_ref_abs_mean_stack)
    G_null_real=G_null.real
    G_null_imag=G_null.imag
    
    dof_null_real=len(av_mod_arr)-1
    dof_null_imag=len(av_mod_arr)-1

    chi_real_null=chis.chi_square(np.array(G_real), np.array(G_null_real), np.array(dG))
    chi_imag_null=chis.chi_square(np.array(G_imag), np.array(G_null_imag), np.array(dG))

    "Constant polarised flux null hypothesis"

    G_null_cpf_real=cs_ref_abs_mean_stack/J
    G_null_cpf_imag=0

    dof_null_real_cpf=len(av_mod_arr)
    dof_null_imag_cpf=len(av_mod_arr)

    chi_real_null_cpf=chis.chi_square(np.array(G_real), np.array(G_null_cpf_real), np.array(dG))
    chi_imag_null_cpf=chis.chi_square(np.array(G_imag), np.array(G_null_cpf_imag), np.array(dG))


    "Performing F-test on the null hypothesis models"


    F = ft.F_test(chi_real_null, chi_imag_null, dof_null_real, dof_null_imag, model_chi_real, model_chi_imag, dof_model_real, dof_model_imag)
    print('pol var',F)

    F_cpf=ft.F_test(chi_real_null_cpf, chi_imag_null_cpf, dof_null_real_cpf, dof_null_imag_cpf, model_chi_real, model_chi_imag, dof_model_real, dof_model_imag)
    print('constant polarised flux',F_cpf)

    F_nulls=ft.F_test(chi_real_null_cpf,chi_imag_null_cpf, dof_null_real_cpf, dof_null_imag_cpf, chi_real_null, chi_imag_null, dof_null_real, dof_null_imag)
    print('F_nulls',F_nulls)


    plt.figure()
    plt.errorbar(av_mod_arr, G_real, yerr=dG, fmt='o', label='G_real')
    plt.plot(av_mod_arr, G_null_real,  label='G_null_real')
    plt.plot(av_mod_arr, fit_y_model_real, label='fit_sin180_real')
    plt.plot(av_mod_arr,[G_null_cpf_real]*len(av_mod_arr), label='G_null_real_cpf')
    plt.legend()
    plt.show()

    plt.figure()
    plt.errorbar(av_mod_arr, G_imag, yerr=dG, fmt='o', label='G_imag')
    plt.plot(av_mod_arr, G_null_imag, label='G_null_imag')
    plt.plot(av_mod_arr, fit_y_model_imag, label='fit_sin180_imag')
    plt.plot(av_mod_arr,[G_null_cpf_imag]*len(av_mod_arr), label='G_null_imag_cpf')
    plt.legend()
    plt.show()
    print('G_real',G_real)
    print('G_imag',G_imag)
    print('G_null_real',G_null_real)
    print('G_null_imag',G_null_imag)
    print('G_null_cpf_real',G_null_cpf_real)
    print(np.repeat(G_null_cpf_real, len(av_mod_arr)))
    print('G_null_cpf_imag',G_null_cpf_imag)
    print('dG',dG)

    # Save results to file
    results = np.column_stack([
    av_mod_arr,
    G_real,
    G_imag,
    G_null_real,
    G_null_imag,
    np.repeat(G_null_cpf_real, len(av_mod_arr)),
    np.repeat(G_null_cpf_imag, len(av_mod_arr)),    
    dG,
    fit_y_model_real,
    fit_y_model_imag
])

    np.savetxt(output_file, results)

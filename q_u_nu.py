#Calculating stokes parameters Q and U as a function of 
#fourier frequency across a range of modulation angles


import numpy as np
from stingray import Lightcurve,Powerspectrum,AveragedCrossspectrum
import sys
sys.path.append('/home/c2032014/py_files')
import load_and_clean as lac
import cross_spec_models as csm
import matplotlib.pyplot as plt
import G_span as gs
import dG_span as dgs
import importlib
importlib.reload(gs)
importlib.reload(csm)
from scipy.optimize import curve_fit



def Q_U_NU(file1,file2,Pmin,Pmax,gti,bin_length,seg_length,fmin,fmax,
           f_bin_number,mod_min,mod_max,mod_bin_number,J,
           spur_sub,coherence_corrector):
    
    
    #Make frequency array
    fspace = np.linspace(fmin, fmax, f_bin_number + 1)
    f_angle_list=[(fspace[i-1],fspace[i]) for i in range(1,len(fspace))] 
    f_min_array = fspace[:-1]
    f_max_array = fspace[1:]
    av_f = (f_min_array + f_max_array) / 2
    av_f_err = (f_max_array - f_min_array) / 2


    #Make modulation angle array
    aspace = np.linspace(mod_min, mod_max, mod_bin_number + 1)
    mod_min_array = aspace[:-1]
    mod_max_array = aspace[1:]
    av_mod = (mod_min_array + mod_max_array) / 2
    av_mod_err = (mod_max_array - mod_min_array) / 2

    #Loading GTI
    GTI=list(np.loadtxt(str(gti)))

    #Loading and cleaning data
    data_1,header_1,*_=lac.load_and_clean(file1,Pmin,Pmax)
    data_2,header_2,*_=lac.load_and_clean(file2,Pmin,Pmax)

    print(data_1)

    #All mod bins file 1 lc
    lc_1_ref=Lightcurve.make_lightcurve(data_1['TIME'],dt=bin_length,gti=GTI)
    lc_1_ref.apply_gtis()

    #All mod bins file 2 lc
    lc_2_ref=Lightcurve.make_lightcurve(data_2['TIME'],dt=bin_length,gti=GTI)
    lc_2_ref.apply_gtis()

    plt.figure()
    plt.title('lc 2 ref')
    plt.plot(lc_2_ref.time,lc_2_ref.counts)
    plt.show()

    #Required fourier products
    ps_2_ref=Powerspectrum.from_lightcurve(lc_2_ref,seg_length,norm='abs')
    cs_ref=AveragedCrossspectrum.from_lightcurve(lc_1_ref,lc_2_ref,seg_length,norm='abs')
    
    #Defining stokes parameters

    I=len(data_1['TIME'])+len(data_2['TIME'])
    Q_norm=(np.sum(data_1['Q'])+np.sum(data_2['Q']))/I
    U_norm=(np.sum(data_1['U'])+np.sum(data_2['U']))/I



    #Calculate G and dG outside the loop and use the cs to calculate the G in the freq required
    G_real_span, G_im_span, n_span, m_span,lc_1_sub_span,lc_spur,cs,spur_sub_norm=gs.G_span(mod_bin_number,data_1,lc_2_ref,GTI,bin_length,seg_length,fmin,fmax,spur_sub,norm='frac')
    

    #dG_span= dgs.dG_span(G_real_span,G_im_span,lc_1_sub_span,n_span,
    #            m_span,f_min,f_max,seg_length,
    #            ps_2_ref_mean,cs_ref_real_mean,coherence_corrector,norm)
        
    plt.figure()
    plt.plot(av_mod,G_real_span,'.')
    plt.show()

    G_real_span_pulse=[i.power.real[(1<=i.freq) & (i.freq<=2)].mean() for i in cs]
    G_real_span_pulse=np.array(G_real_span_pulse)
    G_real_span_pulse=spur_sub_norm*G_real_span_pulse

    plt.figure()
    plt.plot(av_mod,G_real_span_pulse,'.')
    plt.show()
    

    #so the real part is some constant M*C_nu_mag_sqrd and the im part is 0

    #Define arrays to store results
    A_real=[]
    B_real=[]
    C_real=[]

    B_imag=[]
    C_imag=[]

    G_null_real_arr=[]
    G_null_imag_arr=[]

    #Iterating over each frequency bin

    for i in range(len(f_angle_list)):
        
        f_min=f_angle_list[i][0]
        f_max=f_angle_list[i][1]
        print(f_min,f_max)


        G_real_span=[i.power.real[(f_min<=i.freq) & (i.freq<=f_max)].mean() for i in cs]
        G_real_span=np.array(G_real_span)
        G_real_span=spur_sub_norm*G_real_span


        G_im_span=[i.power.imag[(f_min<=i.freq) & (i.freq<=f_max)].mean() for i in cs]
        G_im_span=np.array(G_im_span)
        G_im_span=spur_sub_norm*G_im_span

        #G_im_span =cs.power.imag[(fmin <= cs.freq) & (cs.freq <= fmax)].mean()
        #G_im_span=normalisation*G_im
        
        ps_2_ref_mean=ps_2_ref.power[(f_min<=ps_2_ref.freq) & (ps_2_ref.freq<=f_max)].mean()
        #print('ps_2_ref_mean',ps_2_ref_mean)

        cs_ref_real_mean=np.array(cs_ref.power.real[(f_min<=cs_ref.freq) & (cs_ref.freq<=f_max)].mean())
        cs_ref_im_mean=np.array(cs_ref.power.imag[(f_min<=cs_ref.freq) & (cs_ref.freq<=f_max)].mean()) 
        
        cs_ref_complex_mean=cs_ref_real_mean+1j*cs_ref_im_mean
        cs_ref_abs_mean=np.abs(cs_ref_complex_mean)
        print('cs_ref_abs_mean',cs_ref_abs_mean)
        
        cs_nu_mag_sqrd=cs_ref_abs_mean**2
        print('cs nu mag sqrd',cs_nu_mag_sqrd)
        G_null=csm.cross_spec_model_null(J,Q_norm,U_norm,np.array(av_mod),cs_nu_mag_sqrd)
        
        G_null_real=G_null.real
        #print('G_null_real',G_null_real)
        G_null_real_arr.append(G_null_real)
        G_null_imag=G_null.imag
        #print('G_null_imag',G_null_imag)
        G_null_imag_arr.append(G_null_imag)


        print('cs_ref_real_mean ie A:',cs_ref_real_mean) #same as A within errors


        norm_factor=np.sqrt(cs_ref.power.real[(f_min<=cs_ref.freq) & (cs_ref.freq<=f_max)].mean())
        
        #norm='abs'
    
        
        
        #Fit real and im parts of G
        parameters_real,*_=curve_fit(csm.cross_spec_model_real,np.array(av_mod),np.array(G_real_span))
        A_real.append(parameters_real[0])
        print('A',parameters_real[0])
        B_real.append(parameters_real[1])
        C_real.append(parameters_real[2])
        fit_real = csm.cross_spec_model_real(J,np.array(av_mod),parameters_real[0],parameters_real[1],parameters_real[2])

        parameters_imag,*_=curve_fit(csm.cross_spec_model_imag,np.array(av_mod),np.array(G_im_span))
        B_imag.append(parameters_imag[0])
        C_imag.append(parameters_imag[1])
        fit_imag = csm.cross_spec_model_imag(J,np.array(av_mod),parameters_imag[0],parameters_imag[1])
#        print('PLotting...')
        plt.figure()
        plt.title('Real')
        plt.errorbar(av_mod,G_real_span)#xerr=av_mod_err,yerr=dG_span,ls='None',label='Real')
        #plt.plot(av_mod,G_null_real,label='Null')
        #plt.plot(av_mod,fit_real,label='fit')
        plt.legend()
        plt.show()
        plt.figure()
        plt.title('Imaginary')
        plt.errorbar(av_mod,G_im_span,xerr=av_mod_err)#,yerr=dG_span,ls='None',label='Imaginary')
        #plt.plot(av_mod,fit_imag,label='fit')
        #plt.plot(av_mod,G_null_imag,label='Null')
        plt.legend()
        plt.show()


    #Plot results

    plt.figure()
    plt.title('Re[G] (null)')
    plt.ylabel('Re[G] (null)')
    plt.plot(av_f,G_null_real_arr,'.')
    plt.xlabel('Frequency (Hz)')
    plt.xscale('log')
    plt.yscale('log')
    plt.show()

#    plt.figure()
#    plt.title('Real')
#    plt.plot(av_f,A_real,label='A')
#    plt.plot(av_f,B_real,label='B')
#    plt.plot(av_f,C_real,label='C')
#    plt.legend()

#    plt.figure()
#    plt.title('Imaginary')
#    plt.plot(av_f,B_imag,label='B')
#    plt.plot(av_f,C_imag,label='C')
#    plt.legend()

    return A_real,B_real,C_real,B_imag,C_imag, av_f,norm_factor,G_null_real_arr,G_null_imag_arr

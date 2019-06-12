from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve




def omega_c(B,m): # computes cyclotron frequency
    # B: local magnetic field
    # m: particle mass
    return e*B/m

def omega_p(n,m): # computes plasma frequency
    # n: local density
    # m: particle mass
    return np.sqrt(n*e**2/(m*eps_0))

def omega_cutoff(n,m_e,Omega): # computes theoretical cutoff angular frequency
    # n: local density
    # m_e: electron mass
    # Omega: pulsar angular frequency
    return np.power(2*omega_p(n,m_e)**2*Omega,1./3.)

def chi_para_bar(n_e,n_p,m_e,m_p,omega_loc): # computes rest frame parallel susceptibility
    # n_e: local electron density
    # n_p: local positron density
    # m_e: electron mass
    # m_p: proton mass
    # omega_loc: doppler shifted wave angular frequency
    return -np.divide(omega_p(n_e,m_e)**2,np.power(omega_loc,2))-np.divide(omega_p(n_p,m_p)**2,np.power(omega_loc,2))

def chi_perp_bar(B,n_e,n_p,m_e,m_p,omega_loc): # computes rest frame perpendicular susceptibility
    # B: local magnetic field
    # n_e: local electron density
    # n_p: local positron density
    # m_e: electron mass
    # m_p: proton mass
    # omega_loc: doppler shifted wave angular frequency
    return np.divide(omega_p(n_e,m_e)**2,(omega_c(B,m_e)**2-np.power(omega_loc,2)))+np.divide(omega_p(n_p,m_p)**2,(omega_c(B,m_p)**2-np.power(omega_loc,2)))

def chi_cross_bar(B,n_e,n_p,m_e,m_p,omega_loc): # computes rest frame cross field susceptibility
    # B: local magnetic field
    # n_e: local electron density
    # n_p: local positron density
    # m_e: electron mass
    # m_p: proton mass
    # omega_loc: doppler shifted wave angular frequency
    return np.divide(omega_c(B,m_e),omega_loc)*np.divide(omega_p(n_e,m_e)**2,(omega_c(B,m_e)**2-np.power(omega_loc,2)))-np.divide(omega_c(B,m_p),omega_loc)*np.divide(omega_p(n_p,m_p)**2,(omega_c(B,m_p)**2-np.power(omega_loc,2)))

def chi_para_bar_QED(B,n_e,n_p,m_e,m_p,omega_loc,k_loc): # computes rest frame parallel susceptibility with QED corrections
    # n_e: local electron density
    # n_p: local positron density
    # m_e: electron mass
    # m_p: proton mass
    # omega_loc: doppler shifted wave angular frequency
    # k_loc: wave number
    m_e_0 = np.sqrt(m_e**2+e*B*kappa)
    m_p_0 = np.sqrt(m_p**2+e*B*kappa)
    return -m_e/m_e_0*omega_p(n_e,m_e)**2*np.divide((np.power(omega_loc,2)-k_loc**2*c**2-4*m_e_0**2/kappa**2),(np.power((np.power(omega_loc,2)-k_loc**2*c**2),2)-4*m_e_0**2/kappa**2*np.power(omega_loc,2)))-\
        m_p/m_p_0*omega_p(n_p,m_p)**2*np.divide((np.power(omega_loc,2)-k_loc**2*c**2-4*m_p_0**2/kappa**2),(np.power((np.power(omega_loc,2)-k_loc**2*c**2),2)-4*m_p_0**2/kappa**2*np.power(omega_loc,2)))


def chi_perp_bar_QED(B,n_e,n_p,m_e,m_p,omega_loc,k_loc): # computes rest frame parallel susceptibility with QED corrections
    # B: local magnetic field
    # n_e: local electron density
    # n_p: local positron density
    # m_e: electron mass
    # m_p: proton mass
    # omega_loc: doppler shifted wave angular frequency
    # k_loc: wave number
    m_e_0 = np.sqrt(m_e**2+e*B*kappa)
    m_p_0 = np.sqrt(m_p**2+e*B*kappa)
    return  -m_e/m_e_0*np.divide(omega_p(n_e,m_e)**2,np.power(omega_loc,2))*np.divide((np.power((np.power(omega_loc,2)-k_loc**2*c**2),2)-2*(np.power(omega_loc,2)-k_loc**2*c**2)*m_e/kappa*omega_c(B,m_e)-4*m_e_0**2/kappa**2*np.power(omega_loc,2)),(np.power((np.power(omega_loc,2)-k_loc**2*c**2-2*m_e/kappa*omega_c(B,m_e)),2)-4*m_e_0**2/kappa**2*np.power(omega_loc,2)))-\
        m_p/m_p_0*np.divide(omega_p(n_p,m_p)**2,np.power(omega_loc,2))*np.divide((np.power((np.power(omega_loc,2)-k_loc**2*c**2),2)-2*(np.power(omega_loc,2)-k_loc**2*c**2)*m_p/kappa*omega_c(B,m_p)-4*m_p_0**2/kappa**2*np.power(omega_loc,2)),(np.power((np.power(omega_loc,2)-k_loc**2*c**2-2*m_p/kappa*omega_c(B,m_p)),2)-4*m_p_0**2/kappa**2*np.power(omega_loc,2)))

def chi_cross_bar_QED(B,n_e,n_p,m_e,m_p,omega_loc,k_loc): # computes rest frame parallel susceptibility with QED corrections
    # B: local magnetic field
    # n_e: local electron density
    # n_p: local positron density
    # m_e: electron mass
    # m_p: proton mass
    # omega_loc: doppler shifted wave angular frequency
    # k_loc: wave number
    m_e_0 = np.sqrt(m_e**2+e*B*kappa)
    m_p_0 = np.sqrt(m_p**2+e*B*kappa)
    return np.divide(omega_c(B,m_e),omega_loc)*np.divide(4*m_e**2/kappa**2*omega_p(n_e,m_e)**2,(np.power((np.power(omega_loc,2)-k_loc**2*c**2-2*m_e/kappa*omega_c(B,m_e)),2)-4*m_e_0**2/kappa**2*np.power(omega_loc,2))) - \
        np.divide(omega_c(B,m_p),omega_loc)* np.divide(4*m_p**2/kappa**2*omega_p(n_p,m_p)**2,(np.power((np.power(omega_loc,2)-k_loc**2*c**2-2*m_p/kappa*omega_c(B,m_p)),2)-4*m_p_0**2/kappa**2*np.power(omega_loc,2)))

def n_l(B,n_e,n_p,m_e,m_p,omega_w,Omega): # computes wave index n_l of LCP wave
    # B: local magnetic field
    # n_e: local electron density
    # n_p: local positron density
    # m_e: electron mass
    # m_p: proton mass
    # omega_wave: wave angular frequency in observer frame
    # Omega: pulsar angular frequency
    omega_loc_l = omega_w + Omega # Doppler shifted angular frequency for LCP
    return  np.sqrt(1+chi_perp_bar(B,n_e,n_p,m_e,m_p,omega_loc_l)-chi_cross_bar(B,n_e,n_p,m_e,m_p,omega_loc_l)-np.divide(Omega,omega_w)*(chi_cross_bar(B,n_e,n_p,m_e,m_p,omega_loc_l)-chi_para_bar(n_e,n_p,m_e,m_p,omega_loc_l)-chi_perp_bar(B,n_e,n_p,m_e,m_p,omega_loc_l)))

def n_l_square(B,n_e,n_p,m_e,m_p,omega_w,Omega): # computes wave index n_l of LCP wave squared to test propagation
    # B: local magnetic field
    # n_e: local electron density
    # n_p: local positron density
    # m_e: electron mass
    # m_p: proton mass
    # omega_wave: wave angular frequency in observer frame
    # Omega: pulsar angular frequency
    omega_loc_l = omega_w + Omega # Doppler shifted angular frequency for LCP
    return  1+chi_perp_bar(B,n_e,n_p,m_e,m_p,omega_loc_l)-chi_cross_bar(B,n_e,n_p,m_e,m_p,omega_loc_l)-np.divide(Omega,omega_w)*(chi_cross_bar(B,n_e,n_p,m_e,m_p,omega_loc_l)-chi_para_bar(n_e,n_p,m_e,m_p,omega_loc_l)-chi_perp_bar(B,n_e,n_p,m_e,m_p,omega_loc_l))

def n_r(B,n_e,n_p,m_e,m_p,omega_w,Omega): # computes wave index n_r of RCP wave
    # B: local magnetic field
    # n_e: local electron density
    # n_p: local positron density
    # m_e: electron mass
    # m_p: proton mass
    # omega_wave: wave angular frequency in observer frame
    # Omega: pulsar angular frequency
    omega_loc_r = omega_w - Omega # Doppler shifted angular frequency for RCP
    return  np.sqrt(1+chi_perp_bar(B,n_e,n_p,m_e,m_p,omega_loc_r)+chi_cross_bar(B,n_e,n_p,m_e,m_p,omega_loc_r)-np.divide(Omega,omega_w)*(chi_cross_bar(B,n_e,n_p,m_e,m_p,omega_loc_r)+chi_para_bar(n_e,n_p,m_e,m_p,omega_loc_r)+chi_perp_bar(B,n_e,n_p,m_e,m_p,omega_loc_r)))


def diff_n(B,n_e,n_p,m_e,m_p,omega_w,Omega): # computes wave index difference n_l - n_r
    # B: local magnetic field
    # n_e: local electron density
    # n_p: local positron density
    # m_e: electron mass
    # m_p: proton mass
    # omega_wave: wave angular frequency in observer frame
    # Omega: pulsar angular frequency
    return  n_l(B,n_e,n_p,m_e,m_p,omega_w,Omega)-n_r(B,n_e,n_p,m_e,m_p,omega_w,Omega)


def n_l_QED(B,n_e,n_p,m_e,m_p,omega_w,Omega): # computes wave index n_l of LCP wave with QED corrections
    # B: local magnetic field
    # n_e: local electron density
    # n_p: local positron density
    # m_e: electron mass
    # m_p: proton mass
    # omega_wave: wave angular frequency in observer frame
    # Omega: pulsar angular frequency
    n_l_no_QED = n_l(B,n_e,n_p,m_e,m_p,omega_w,Omega) #wave index without QED corrections
    n_l_QED = n_l_no_QED # initialize a vector for wave index with QED corrections
    for i,W in enumerate(omega_w):
        omega_loc = W + Omega # Doppler shifted angular frequency for LCP
        def funcl(k_loc):
            return k_loc**2*c**2/W**2-(1+chi_perp_bar_QED(B,n_e,n_p,m_e,m_p,omega_loc,k_loc)-chi_cross_bar_QED(B,n_e,n_p,m_e,m_p,omega_loc,k_loc)-Omega/W*(chi_cross_bar_QED(B,n_e,n_p,m_e,m_p,omega_loc,k_loc)-chi_para_bar_QED(B,n_e,n_p,m_e,m_p,omega_loc,k_loc)-chi_perp_bar_QED(B,n_e,n_p,m_e,m_p,omega_loc,k_loc)))
        n_l_QED[i] = fsolve(funcl, n_l_no_QED[i]*W/c)*c/W # solves implicit function for n_l_QED using n_l_no_QED as starting point
    return n_l_QED



def n_r_QED(B,n_e,n_p,m_e,m_p,omega_w,Omega): # computes wave index n_R of RCP wave with QED corrections
    # B: local magnetic field
    # n_e: local electron density
    # n_p: local positron density
    # m_e: electron mass
    # m_p: proton mass
    # omega_wave: wave angular frequency in observer frame
    # Omega: pulsar angular frequency
    n_r_no_QED = n_r(B,n_e,n_p,m_e,m_p,omega_w,Omega) # wave index without QED corrections
    n_r_QED = n_r_no_QED # initialize a vector for wave index with QED corrections
    for i,W in enumerate(omega_w):
        omega_loc = W - Omega # Doppler shifted angular frequency for RCP
        def funcr(k_loc):
            return k_loc**2*c**2/W**2-(1+chi_perp_bar_QED(B,n_e,n_p,m_e,m_p,omega_loc,k_loc)+chi_cross_bar_QED(B,n_e,n_p,m_e,m_p,omega_loc,k_loc)-Omega/W*(chi_cross_bar_QED(B,n_e,n_p,m_e,m_p,omega_loc,k_loc)+chi_para_bar_QED(B,n_e,n_p,m_e,m_p,omega_loc,k_loc)+chi_perp_bar_QED(B,n_e,n_p,m_e,m_p,omega_loc,k_loc)))
        n_r_QED[i] = fsolve(funcr, n_r_no_QED[i]*W/c)*c/W # solves implicit function for n_r_QED using n_l_no_QED as starting point
    return n_r_QED


def diff_n_QED(B,n_e,n_p,m_e,m_p,omega_w,Omega): # computes wave index difference n_l - n_r with QED corrections
    # B: local magnetic field
    # n_e: local electron density
    # n_p: local positron density
    # m_e: electron mass
    # m_p: proton mass
    # omega_wave: wave angular frequency in observer frame
    # Omega: pulsar angular frequency
    return  n_l_QED(B,n_e,n_p,m_e,m_p,omega_w,Omega)-n_r_QED(B,n_e,n_p,m_e,m_p,omega_w,Omega)


def delta_phi(B,n_e,n_p,m_e,m_p,omega_w,Omega): # computes rotary power from wave index difference
    # B: local magnetic field
    # n_e: local electron density
    # n_p: local positron density
    # m_e: electron mass
    # m_p: proton mass
    # omega_wave: wave angular frequency in observer frame
    # Omega: pulsar angular frequency
    return diff_n(B,n_e,n_p,m_e,m_p,omega_w,Omega)*omega_w/(2*c) # rotary power from wave index difference, as per Eq. (3)

def delta_phi_QED(B,n_e,n_p,m_e,m_p,omega_w,Omega): # computes rotary power from wave index difference with QED corrections
    # B: local magnetic field
    # n_e: local electron density
    # n_p: local positron density
    # m_e: electron mass
    # m_p: proton mass
    # omega_wave: wave angular frequency in observer frame
    # Omega: pulsar angular frequency
    return diff_n_QED(B,n_e,n_p,m_e,m_p,omega_w,Omega)*omega_w/(2*c) # rotary power from wave index difference, as per Eq. (3)

def plot_Fig_2(B,n,Omega): # produces Figure 2
    # B: magnetic field
    # n: density
    # Omega: pulsar angular frequency
    omega = np.linspace(1.001*omega_cutoff(n,m_e,Omega),2.5*omega_cutoff(n,m_e,Omega),500) # initialize vector for wave angular frequency in observer rest frame
    delta_phi_asymptotic_0 = -np.divide(omega_p(n,m_e)**2*Omega,np.power(omega,2))/c# rotary power asymptotic solution for frequency much greater than the cut-off, as per Eq. (24)
    delta_phi_asymptotic = (-np.sqrt(2.)+np.sqrt(3.)/np.power(2.,1./6.)*np.sqrt((omega-omega_cutoff(n,m_e,Omega))/(np.power(omega_p(n,m_e),2./3.)*np.power(Omega,1./3.))))*omega/(2*c) # rotary power asymptotic solution near the cut-off, as per Eq. (25)

    #Figure 2
    plt.figure()
    plt.plot(omega/omega_c(B,m_e),delta_phi(B,n,n,m_e,m_p,omega,Omega),'blue') # rotary power for symmetrical e-p plasma from exact solution of Eq. (20)
    plt.plot(omega/omega_c(B,m_e),delta_phi_asymptotic_0,color='#8b0000', linestyle='dashed') # high-frequency asymptotic solution
    plt.plot(omega/omega_c(B,m_e),delta_phi_asymptotic,':g') # near cut-off asymptotic solution
    plt.plot(omega_cutoff(n,m_e,Omega)/omega_c(B,m_e)*np.array([1,1]),np.array([-1,0]),'--k') # cut-off frequency for symmetrical case
    plt.xlim(1.75e-11,5.25e-11) # set x range
    plt.ylim(-0.9,0) # set y range
    plt.xlabel(r'$\omega/\omega_{c}$') # label x axis
    plt.ylabel(r'$\delta^{M}$ [rad.m$^{-1}$]') # label y axis

    a = plt.axes([0.45, 0.2, 0.4, .3])
    plt.plot(omega/omega_c(B,m_e),delta_phi(B,n,n,m_e,m_p,omega,Omega),'blue') # rotary power for symmetrical e-p plasma from exact solution of Eq. (20)
    plt.plot(omega/omega_c(B,m_e),delta_phi_asymptotic_0,color='#8b0000', linestyle='dashed') # high-frequency asymptotic solution
    plt.plot(omega/omega_c(B,m_e),delta_phi_asymptotic,':g') # near cut-off asymptotic solution
    plt.plot(omega_cutoff(n,m_e,Omega)/omega_c(B,m_e)*np.array([1,1]),np.array([-1,0]),'--k') # cut-off frequency for symmetrical case
    plt.xlim(2.15e-11,2.4e-11) # set x range
    plt.ylim(-0.9,-0.6) # set y range
    
    # write data to comma delimited .csv file. 
    #first row: wave angular frequency
    #second row: rotary power
    #third row: high-frequency asymptotic rotary power 
    #fourth row: asymptotic rotary power near cut-off frequency    
    with open("data_Fig_2.csv" , 'w') as f:
        f.write('omega_wave,')
        f.write(", ".join(map(str, omega)))
        f.write("\n")
        f.write('delta,')
        f.write(", ".join(map(str, delta_phi(B,n,n,m_e,m_p,omega,Omega))))
        f.write("\n")
        f.write('delta_HF,')
        f.write(", ".join(map(str, delta_phi_asymptotic_0)))
        f.write("\n")
        f.write('delta_near_cutoff,')
        f.write(", ".join(map(str, delta_phi_asymptotic)))
        
    



def plot_Fig_3(B,n,Omega,h_em): # produces Figure 3
    # B: surface magnetic field
    # n: surface density
    # Omega: pulsar angular frequency
    # h_em: emission height
    RM_ISM = 5 #ad-hoc RM for Faraday rotation in the ISM, in rad.m^-2
    omega_cut_off_h_em = omega_cutoff(n*(r_star/h_em)**3,m_e,Omega) # cut-off frequency at emission height
    omega = np.linspace(1.001*omega_cut_off_h_em,2*np.pi*1.0e7,60) # wave angular frequency vector for tracking change in polarisation as function of angular frequency
    omega = np.append(omega,omega[-1]*1.e2) # add large angular frequency at the end of vector to compute "infinite frequency" MOR integrate deffect
    lambda_w = c*np.divide(2.*np.pi,omega) # wavelength
    f_w = omega/(2*np.pi) # wave frequency
    omega_no_MOR = np.linspace(omega_cut_off_h_em/2.,2*np.pi*1.0e7,90) # larger range wave angular frequency vector for plotting Faraday rotation below cut-off
    lambda_w_no_MOR = c*np.divide(2.*np.pi,omega_no_MOR) # wavelength
    f_w_no_MOR = omega_no_MOR/(2*np.pi) # wave frequency
    z = h_em #photon starts at h_em
    PA_MOR = np.zeros(omega.shape) # PA angle vector to store PA accumulated due to MOR for each angular frequency
    PA_MOR = PA_MOR + 1.e-20 # avoid division by zero for first iteration of while loop
    dt = r_star/(100*c) # time step to advance photon in space and accumulate PA changes
    d_phi = delta_phi(B*(r_star/z)**3,n*(r_star/z)**3,n*(r_star/z)**3,m_e,m_p,omega,Omega)*c*dt # PA change due to MOR in current time step
    while (np.linalg.norm(np.divide(d_phi,PA_MOR))>1e-5): # integrate MOR effects up to a position where the next step contribution is small enough compared to the already accumulated PA
        PA_MOR = PA_MOR + d_phi # update PA with current step contribution
        z = z + c*dt # move test photon
        d_phi = delta_phi(B*(r_star/z)**3,n*(r_star/z)**3,n*(r_star/z)**3,m_e,m_p,omega,Omega)*c*dt # compute PA change due to MOR at new position
    print('MOR effects integrated from %.1f to %.1f neutron star radius' % (h_em/r_star,z/r_star)) # print radius up to which MOR is accounted for for verification purpose

    PA_Faraday_alone = RM_ISM*np.power(lambda_w,2) # PA rotation equivalent to assumed RM in the ISM
    #Figure 3
    plt.figure()
    plt.subplot(211)
    plt.plot(f_w_no_MOR,RM_ISM*np.power(lambda_w_no_MOR,2)/1.e3,'k',label='RM$^{ISM}$') # PA due to Faraday rotation in the ISM alone
    plt.plot(f_w[1::2],(PA_Faraday_alone[1::2]+PA_MOR[1::2])/1.e3,'^r',label='MOR+RM$^{ISM}$') # PA due to both Faraday rotation in the ISM and MOR in rotating magnetosphere
    plt.plot(omega_cut_off_h_em*np.array([1,1])/(2*np.pi),np.array([0,20]),'--k') # cut-off frequency vertical dashed line
    plt.xlim(5.6e6,1e7) # set x range
    plt.ylim(3,15) # set y range
    plt.ylabel(r'PA [$\times 10^3$ rad]') # label y axis
    plt.legend(loc='upper right', bbox_to_anchor=(0.7, 0.95),ncol=1, prop={'size': 10}) # add plot legend

    plt.subplot(212)
    plt.plot(f_w,np.divide(PA_MOR,np.power(lambda_w,2))+RM_ISM, 'blue')
    plt.plot(np.array([5e6,1.5e7]),(PA_MOR[-1]/lambda_w[-1]**2+RM_ISM)*np.array([1,1]),'-.k') # infinite frequency RM horizontal dashed-dotted line
    plt.plot(omega_cut_off_h_em*np.array([1,1])/(2*np.pi),np.array([3,4]),'--k') # cut-off frequency vertical dashed line
    plt.xlim(5.6e6,1e7) # set x range
    plt.ylim(3.61,3.71) # set y range
    plt.xlabel(r'$\omega/2\pi$ [Hz]') # label x axis
    plt.ylabel(r'RM [rad.m$^{-2}$]') # label y axis
    
    
    # write data to comma delimited .csv file. 
    #first row: wave frequency below and above cut-off
    #second row: PA due to Faraday in ISM alone
    #third row: wave frequency above cut-off 
    #fourth row: PA due to Faraday in ISM and MOR in magnetosphere
    #fifth row: RM due to Faraday in ISM and MOR in magnetosphere  
    with open("data_Fig_3.csv" , 'w') as f:
        f.write('f_wave_no_MOR,')
        f.write(", ".join(map(str, f_w_no_MOR)))
        f.write("\n")
        f.write('PA_ISM_alone,')
        f.write(", ".join(map(str, RM_ISM*np.power(lambda_w_no_MOR,2))))
        f.write("\n")
        f.write('f_wave,')
        f.write(", ".join(map(str, f_w)))
        f.write("\n")
        f.write('PA_ISM_and_MOR,')
        f.write(", ".join(map(str, PA_Faraday_alone+PA_MOR)))
        f.write("\n")
        f.write('RM_ISM_and_MOR,')
        f.write(", ".join(map(str, np.divide(PA_MOR,np.power(lambda_w,2))+RM_ISM)))

def plot_Fig_4(B,n,Omega,f_density): # produces Figure 4
    # B: magnetic field
    # n: density
    # Omega: pulsar angular angular frequency
    # f_density : density asymmetry, np = n, ne = (1-f_density)/f_density*n
    omega = np.linspace(1.001*omega_cutoff(n,m_e,Omega),2.5*omega_cutoff(n,m_e,Omega),500) # initialize vector for wave angular frequency in observer rest frame
    omega_asym = omega[np.where(n_l_square(B,(1.-f_density)/f_density*n,n,m_e,m_p,omega,Omega) > 0.0 )] # get subset of vector omega where LCP propagates in the asymmetric case, i. e. omega for which n_l(omega)^2>0
    delta_phi_asymptotic_0 = -np.divide(omega_p(n,m_e)**2*Omega,np.power(omega,2))/c# rotary power asymptotic solution for angular frequency much greater than the cut-off, as per Eq. (24)

    #Figure 4
    plt.figure()
    plt.plot(omega/omega_c(B,m_e),delta_phi(B,n,n,m_e,m_p,omega,Omega),'blue',label='symmetrical') # rotary power for symmetrical e-p plasma from exact solution of Eq. (20)
    plt.plot(omega/omega_c(B,m_e),delta_phi_QED(B,n,n,m_e,m_p,omega,Omega),'--g',label='QED corrected') # rotary power for symmetrical e-p plasma from exact solution of Eq. (20) with QED corrections
    plt.plot(omega/omega_c(B,m_e),delta_phi_asymptotic_0,color='#8b0000', linestyle='dashed') # high-frequency asymptotic solution
    plt.plot(omega_asym/omega_c(B,m_e),delta_phi(B,(1-f_density)/f_density*n,n,m_e,m_p,omega_asym,Omega),'--r',label='non-symmetrical') # rotary power for asymmetrical e-p plasma from exact solution of Eq. (20)
    plt.plot(omega_cutoff(n,m_e,Omega)/omega_c(B,m_e)*np.array([1,1]),np.array([-1,0]),'--k') # cut-off frequency for symmetrical case
    plt.xlim(1.75e-11,5.25e-11) # set x range
    plt.ylim(-0.9,0) # set y range
    plt.xlabel(r'$\omega/\omega_{c}$') # label x axis
    plt.ylabel(r'$\delta^{M}$ [rad.m$^{-1}$]') # label y axis
    plt.legend(loc='upper right', bbox_to_anchor=(0.7, 0.95),ncol=2, prop={'size': 10})  # add plot legend

    a = plt.axes([0.45, 0.2, 0.4, .3])
    plt.plot(omega/omega_c(B,m_e),delta_phi(B,n,n,m_e,m_p,omega,Omega),'blue') # rotary power for symmetrical e-p plasma from exact solution of Eq. (20)
    plt.plot(omega/omega_c(B,m_e),delta_phi_QED(B,n,n,m_e,m_p,omega,Omega),'--g') # rotary power for symmetrical e-p plasma from exact solution of Eq. (20) with QED corrections
    plt.plot(omega/omega_c(B,m_e),delta_phi_asymptotic_0,color='#8b0000', linestyle='dashed') # high-frequency asymptotic solution
    plt.plot(omega_asym/omega_c(B,m_e),delta_phi(B,(1-f_density)/f_density*n,n,m_e,m_p,omega_asym,Omega),'--r') # rotary power for asymmetrical e-p plasma from exact solution of Eq. (20)
    plt.plot(omega_cutoff(n,m_e,Omega)/omega_c(B,m_e)*np.array([1,1]),np.array([-1,0]),'--k') # cut-off frequency for symmetrical case
    plt.xlim(2.15e-11,2.4e-11) # set x range
    plt.ylim(-0.9,-0.6) # set y range

    
    # write data to comma delimited .csv file. 
    #first row: wave angular frequency
    #second row: rotary power
    #third row: rotary power with QED corrections
    #fourth row: high-frequency asymptotic rotary power
    #fifth row: wave angular frequency fro asymmetric density case
    #sixth row: rotary power with density asymmetry  
    with open("data_Fig_4.csv" , 'w') as f:
        f.write('omega_wave,')
        f.write(", ".join(map(str, omega)))
        f.write("\n")
        f.write('delta,')
        f.write(", ".join(map(str, delta_phi(B,n,n,m_e,m_p,omega,Omega))))
        f.write("\n")
        f.write('delta_with_QED,')
        f.write(", ".join(map(str, delta_phi_QED(B,n,n,m_e,m_p,omega,Omega))))
        f.write("\n")
        f.write('delta_HF,')
        f.write(", ".join(map(str, delta_phi_asymptotic_0)))
        f.write("\n")
        f.write('omega_wave_asym,')
        f.write(", ".join(map(str, omega_asym)))
        f.write("\n")
        f.write('delta_with_density_asym,')
        f.write(", ".join(map(str, delta_phi(B,(1-f_density)/f_density*n,n,m_e,m_p,omega_asym,Omega))))



# constants
e = 1.60217662e-19 # electron charge
eps_0 = 8.854187e-12 # vacuum permittivity
m_e = 9.10938356e-31 # electron mass
c = 2.99792458e8 # speed of light
h = 6.62607e-34 # Planck constant
hbar = h/(2*np.pi) # Planck constant divided by 2 pi = hbar
kappa = hbar/c**2 # hbar/c^2

# canonical normal pulsar parameters as defined in Table 1
B0 = 10.0**8 # surface magnetic field
n0 = 7.0*10**20 # surface density
Omega = 2*np.pi/0.5 # pulsar angular frequency
m_p = m_e # e-p plasma
r_star = 10.e3 # neutron star radius
f = 0.49 # density asymmetry
h_em = 10.*r_star # emission height



Fig2 = plot_Fig_2(B0,n0,Omega)
Fig3 = plot_Fig_3(B0,n0,Omega,h_em)
Fig4 = plot_Fig_4(B0,n0,Omega,f)
plt.show()

# packages
import numpy as np
import scipy.integrate as integrate
import scipy.optimize as opt
import matplotlib.pyplot as plt
import csv
from astropy.cosmology import FlatLambdaCDM as magic

plt.style.use("dark_background")

# constant
c = 2.99792458 * 1e5 # units: km/s

def D_L(z, omega_m, omega_lambda, H_0):
    omega_k = 1-omega_m-omega_lambda
    if (omega_k == 0): # handle this case separately to avoid division by zero errors
        def integrand(x):
            return 1/np.sqrt((1+x)**2*(1+omega_m*x) - x*(2+x)*omega_lambda)
        integral = integrate.quad(integrand,0,z)[0]
        r = c*integral/H_0
        return (1+z)*r
        
    prefactor = (c/H_0) * (1+z) * (1/np.sqrt(np.abs(omega_k)))
    def integrand(x):
        return 1/np.sqrt((1+x)**2 * (1+omega_m*x) - x * (2+x) * omega_lambda)
    integral = integrate.quad(integrand,0,z)[0]
    arg = np.sqrt(np.abs(omega_k)) * integral
    sinn = np.sin(arg) if omega_k <= 0 else np.sinh(arg)
    return prefactor * sinn

def D_L_helper(redshifts, omega_m, omega_lambda, H_0):
    return [D_L(z,omega_m,omega_lambda,H_0) for z in redshifts]

def scipy_fit(redshift, lum_dist, lum_dist_err):
    bnds = ((0,0,60),(2,2,80))
    popt, pcov = opt.curve_fit(D_L_helper, redshift, lum_dist, p0=[0.3,0.7,70], bounds = bnds, sigma = lum_dist_err)
    omega_m, omega_lambda, H_0 = popt
    omega_m_err, omega_lambda_err, H_0_err = np.sqrt(np.diag(pcov))
    return omega_m, omega_lambda, H_0, omega_m_err, omega_lambda_err, H_0_err

def scipy_fit_plot(redshift, lum_dist, lum_dist_err, omega_m, omega_lambda, H_0):
    plt.figure(figsize = (20,15))
    zz = np.linspace(np.min(redshift),np.max(redshift),1000)
    plt.errorbar(np.log(redshift),lum_dist,yerr=lum_dist_err,fmt='.',color='purple')
    plt.errorbar(np.log(zz),D_L_helper(zz, omega_m, omega_lambda, H_0), linewidth=3,color='white')
    plt.xlabel("$\log(z)$")
    plt.ylabel("$D_L$ (Mpc)")
    plt.legend(("Union Data","Best Fit Model"),loc='upper left')
    plt.title("Luminosity Distance vs. Redshift (SCP)")
    plt.show()
    
def phase_space_plots(redshift, omega_m, omega_lambda, H_0, lum_dist, lum_dist_err):
    num_points = 20

    omega_m_range = np.linspace(0,1.5,num_points)
    omega_lambda_range = np.linspace(0,1.5,num_points)

    H_0_range = np.linspace(60,80,num_points)

    omega_m_best = omega_m_range[0]
    omega_lambda_best = omega_lambda_range[0]
    H_0_best = H_0_range[0]
    expected = D_L_helper(redshift, omega_m, omega_lambda, H_0)
    chi2_best = np.sum((lum_dist - expected)**2/lum_dist_err**2)
    k_best = 0


    chi2_arr = np.zeros((num_points,num_points,num_points))
    new_array = np.zeros((num_points,num_points))

    for k in np.arange(np.size(H_0_range)):
        for i in np.arange(np.size(omega_m_range)):
            for j in np.arange(np.size(omega_lambda_range)):
                    omega_m = omega_m_range[i]
                    omega_lambda = omega_lambda_range[j]
                    H_0 = H_0_range[k]
                    expected = D_L_helper(redshift, omega_m, omega_lambda, H_0)
                    chi2 = np.sum((lum_dist - expected)**2/lum_dist_err**2)
                    chi2_arr[i][j][k] = chi2
                    new_array[i][j] = chi2
                    if chi2 < chi2_best:
                        omega_m_best = omega_m
                        omega_lambda_best = omega_lambda
                        H_0_best = H_0
                        chi2_best = chi2
                        k_best = k

    print("Matter density: ", "{0:.2f}".format(omega_m_best))
    print("Dark energy density: ", "{0:.2f}".format(omega_lambda_best))
    print("Hubble constant: ", "{0:.2f}".format(H_0_best))
    chi2_slice = chi2_arr[:][:][k_best]
    plt.figure(figsize = (20,15))
    im = plt.imshow(chi2_slice, interpolation='bilinear', origin='lower', \
                    cmap='seismic', extent=(0.,1.0,0.,1.5))
    plt.colorbar()
    #plt.plot(X[am],Y[am],'r*',markersize=20)
    plt.xlabel(r'$\Omega_M$')
    plt.ylabel(r'$\Omega_\Lambda$')
    plt.title('Brute Force Reduced Chi-Square-Fit (SCP Data)')
    plt.plot(omega_m_best,omega_lambda_best,'r*',markersize=20)
    plt.show()

def astropy_plots(H_0_best, omega_m_best):
    cosmo = magic(H0=H_0_best, Om0=omega_m_best)
    plt.figure(figsize = (25,15))
    scale_factor = cosmo.scale_factor(redshift)
    plt.scatter(redshift, scale_factor)
    plt.xlabel('Redshift')
    plt.ylabel('Scale Factor')
    plt.title('Scale Factor vs Redshift (SCP Data)')
    plt.show()

    plt.figure(figsize = (25,15))
    h = cosmo.H(redshift)
    plt.scatter(redshift, h)
    plt.xlabel('Redshift')
    plt.ylabel('Hubble Parameter')
    plt.title('Hubble Parameter vs Redshift (SCP Data)')
    plt.show()

    plt.figure(figsize = (25,15))
    zz = np.linspace(np.min(redshift),np.max(redshift),1000)
    plt.errorbar(np.log(redshift),lum_dist,yerr=lum_dist_err,fmt='.',color='purple')
    plt.errorbar(np.log(zz),D_L_helper(zz, omega_m_best, omega_lambda_best, H_0_best), linewidth=3,color='white')
    plt.xlabel("$\log(z)$")
    plt.ylabel("$D_L$ (Mpc)")
    plt.legend(("Union Data","Best Fit Model"),loc='upper left')
    plt.title("Luminosity Distance vs. Redshift (SCP)");
    plt.show()
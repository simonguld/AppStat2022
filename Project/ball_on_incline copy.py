# Author: Simon Guldager Andersen
# Date (latest update): 12/12-2022

### SETUP -----------------------------------------------------------------------------------------------------------------------------------

## Imports:
import numpy as np
import os, sys, itertools
import iminuit as Minuit
from scipy import stats, optimize
import matplotlib.pyplot as plt



sys.path.append('C:\\Users\\Simon\\PycharmProjects\\Projects\\Projects\\AppStat2022\\External_Functions')
from ExternalFunctions import Chi2Regression, BinnedLH
from ExternalFunctions import nice_string_output, add_text_to_ax # useful functions to print fit results on figure

## Change directory to current one
os.chdir('Projects\\AppStat2022\\Project')


### FUNCTIONS -------------------------------------------------------------------------------------------------------------------------------

def extract_peak_times(time_index_arr, n_peaks, cluster_val):
    """
    Take an array of times indices corresponding to all times just before and after a peak as determined by some tolerance.
    Remove all but 1 time index at each side of each peak. This is done by removining all indices within cluster_val of the
    chosen index. The indices are chosen as the smallest index of the left side of a peak and the largest index on the right side
    for maximal symmetry. n_peaks is the number of peaks for which to carry out this procedure
    An array containing one left and right endpoint of each peak is returned
    """
    # copy array of time indices
    time_indices = time_index_arr.astype('int')
    for i in range(2 * n_peaks):
        # find minimum in remaning list:
        val = np.min(time_indices[i:])
        if i % 2 != 0:
            # For unequal i, the right side of a peak, we need the biggest index (since we take the smallest index for equal i)
            # comparing to cluster_val makes sure the only values on one side of a peak is considered
            index = np.argwhere(np.abs(time_indices - val) < cluster_val).flatten()
            val = np.max(time_indices[index])
        # Remove indices corresponding to same side of peak as val
        index = np.argwhere((np.abs(time_indices - val) > cluster_val) | (time_indices == val)).flatten()
        time_indices = time_indices[index]

    return time_indices

def extract_peak_times_mod(voltage_arr, time_index_arr, half_height, n_peaks, cluster_val):
    """
    Take an array of times indices corresponding to all times just before and after a peak as determined by some tolerance.
    Remove all but 1 time index at each side of each peak. This is done by removining all indices within cluster_val of the
    chosen index. The indices are chosen as the smallest index of the left side of a peak and the largest index on the right side
    for maximal symmetry. n_peaks is the number of peaks for which to carry out this procedure
    An array containing one left and right endpoint of each peak is returned
    """
    # copy array of time indices
    time_indices = time_index_arr.astype('int')
    index_start = 0
    for i in range(n_peaks):
        # find minimum in remaning list:
        index_min = np.min(time_indices[i:])

        extract_peak_vals = np.argwhere((np.abs(time_indices - index_min) < cluster_val) | (time_indices == index_min)).flatten()
    
        best_val = np.argmin(np.abs(voltage_arr[time_indices[extract_peak_vals]] - half_height)).flatten()
        best_val = best_val + index_min
   
    
        # Remove indices corresponding to same side of peak as val
        index = np.argwhere((np.abs(time_indices - best_val) > 15 * cluster_val) | (time_indices == best_val)).flatten()
        time_indices = time_indices[index]
    return time_indices

def quad_fit(t, a, v0, s0):
    return 0.5 * a * t ** 2 + v0 * t + s0 

def chi2_func(list, list_err):
    av = np.mean(list)
    chi2_val = np.sum(np.power((list-av),2) / np.power(list_err, 2))
    return chi2_val

def calc_dtheta(a_norm, a_rev, theta):
    return ((a_norm - a_rev) * np.sin(theta)) / ((a_norm + a_rev) * np.cos(theta))

def calc_error_dtheta(a_norm, a_rev, theta, err_a_norm, err_a_rev, err_theta):
    ddtheta_da_norm = (1 - (a_norm - a_rev) / (a_norm + a_rev)   ) * np.tan(theta) / (a_norm + a_rev)
    ddtheta_da_rev = - (1 + (a_norm - a_rev) / (a_norm + a_rev)   ) * np.tan(theta) / (a_norm + a_rev)
    ddtheta_dtheta = (a_norm - a_rev) / ((a_norm + a_rev) * np.cos(theta) ** 2 )

    return np.sqrt(ddtheta_da_norm ** 2 * err_a_norm ** 2 + ddtheta_da_rev ** 2 * err_a_rev ** 2 + ddtheta_dtheta **2 * err_theta ** 2)

### MAIN ------------------------------------------------------------------------------------------------------------------------------------

# Set parameters
plot1 = True 
# The angle in degrees. Convert
theta = 13.282 * (np.pi / 180)
theta_err = 0.182 * (np.pi / 180)

# set style
plt.style.use('seaborn')


# The idea is to extract the times at which the ball passes the 5 gauges. Combined with the masurements 
# of the relative distances of the gauges, a fit can be performed to estimate the acceleration, 
# where the uncertainty on t and s must be taken into account

# Now, doing this for both the direct and reverse orientation gives an estimate of the angle between the table and the ground

def main():
    
    ## Load data
    norm_big = [None] * 4
    norm_small = [None] * 4
    rev_big = [None] * 4
    rev_small = [None] * 4

    names = ['simon', 'maxi', 'masoumeh', 'yasmine']

    for i, name in enumerate(names):
        norm_big[i] =  (np.genfromtxt(f'big_ball_{name}.csv', delimiter=','))[:,0:2].T
        norm_small[i] =  (np.genfromtxt(f'small_ball_{name}.csv', delimiter=','))[:,0:2].T
        rev_big[i] =  (np.genfromtxt(f'rev_big_ball_{name}.csv', delimiter=','))[:,0:2].T
        rev_small[i] =  (np.genfromtxt(f'rev_small_ball_{name}.csv', delimiter=','))[:,0:2].T
    data_list = [norm_big, rev_big, norm_small, rev_small]


 

    #clean data: Inspection of the plots reveal an additional broad peak in data_rev after the 5 gauge peaks. It is unclear
    # what this means, but we take it to be noise. All peaks occur before t = 1s, and be restrict our attention to values below
    # this cutoff

    # Extract middle points
    peak = 5 
    offset = 2
    half_height = 0.5 * (peak + offset)

    n_peaks = 5
    cluster_val = 20

    half_height_BB = [None] * 4
    half_height_BBr = [None] * 4
    half_height_SB = [None] * 4
    half_height_SBr = [None] * 4
    half_height_list = [half_height_BB, half_height_BBr, half_height_SB, half_height_SBr]

    for i, data_set in enumerate(data_list):
        half_heights = half_height_list[i]
        for j, data in enumerate(data_set):
            indices = (np.argwhere(np.abs(data[1,:] - half_height) < 0.5).flatten()).astype('int')
            half_heights[j] = extract_peak_times_mod(data[1], indices, half_height = half_height, n_peaks = n_peaks, cluster_val = 20)

    size = (6,10)
    exp_names = ['big ball direct orientation', 'big ball rev. orientation', 'small ball dir. orientation', 'small ball rev.orientation']
    if plot1:
        # Plot and fit data data
       
        fig_BB, ax_BB = plt.subplots(ncols = 4, figsize = size)
        fig_BBr, ax_BBr = plt.subplots(ncols = 4, figsize = size)
        fig_SB, ax_SB = plt.subplots(ncols = 4, figsize = size)
        fig_SBr, ax_SBr = plt.subplots(ncols = 4, figsize = size)
        ax_list = [ax_BB.flatten(), ax_BBr.flatten(), ax_SB.flatten(), ax_SBr.flatten()]
        fig_list = [fig_BB, fig_BBr, fig_SB, fig_SBr]
        
 
    
        for j, data_set in enumerate(data_list):
            ax = ax_list[j]
            half_heights = half_height_list[j]
            fig_list[j].suptitle(f'Voltage fluctuation for {exp_names[j]}')

            #Print times to file
            original = sys.stdout
            with open(f'times_{names[j]}.dat', 'w' ) as f: 

                for i, data in enumerate(data_set): 

                    ax[i].plot(data[0,:], data[1,:], 'k.-')
                    ax[i].plot(data[0,half_heights[i]], data[1,half_heights[i]], 'rx',  markersize = 8,  mew = 2)

                    xmin, xmax = data[0, half_heights[i][0]], data[0, half_heights[i][-1]]
                    ax[i].set( xlabel = 'Time (s)', ylabel = 'Voltage (V)', xlim = (xmin - 0.1, xmax + 0.1), ylim = (1.5,5.5))


                    sys.stdout = f
            
                    print(data[0,half_heights[i]])
            
                fig_list[j].tight_layout()

            sys.stdout = original

        plt.show()


    # fit distance time measurements to extract accelerations

# 186.543
    distances = np.array([187.5, 364.736, 545.614, 734.15, 911.357]) / 1000

    distances_err = np.array([0.5, 0.7071067812, 0.25,0.4082482905, 0.25]) / 1000

    fig_BBf, ax_BBf = plt.subplots(ncols = 4, figsize = size)
    fig_BBrf, ax_BBrf = plt.subplots(ncols = 4, figsize = size)
    fig_SBf, ax_SBf = plt.subplots(ncols = 4, figsize = size)
    fig_SBrf, ax_SBrf = plt.subplots(ncols = 4, figsize = size)
    ax_listf = [ax_BBf.flatten(), ax_BBrf.flatten(), ax_SBf.flatten(), ax_SBrf.flatten()]
    fig_listf = [fig_BBf, fig_BBrf, fig_SBf, fig_SBrf]
    exp_names = ['big ball direct orientation', 'big ball rev. orientation', 'small ball dir. orientation', 'small ball rev. orientation']
    a_BB_norm = [None] * 4
    a_BB_rev = [None] * 4
    a_SB_norm = [None] * 4
    a_SB_rev = [None] * 4
    aerr_BB_norm = [None] * 4
    aerr_BB_rev = [None] * 4
    aerr_SB_norm = [None] * 4
    aerr_SB_rev = [None] * 4
    a_list = [a_BB_norm, a_BB_rev, a_SB_norm, a_SB_rev]
    aerr_list = [aerr_BB_norm, aerr_BB_rev, aerr_SB_norm, aerr_SB_rev]

    for j, data_set in enumerate(data_list):
        ax = ax_listf[j]
        fig = fig_listf[j]
        half_heights = half_height_list[j]
        fig_listf[j].suptitle(f'Displacement for {exp_names[j]} with fit  y = 0.5*a*t^2 + v0*t + s0')
        a_vals = a_list[j]
        a_errors = aerr_list[j]
        for i, data in enumerate(data_set): 
            times = data[0,half_heights[i]]
            #fig.set_title(f'Displacement for {exp_names[j]} with fit  y = 0.5*a*t^2 + v0*t + s0')
            ax[i].errorbar(times, distances, distances_err,  fmt='.', label='Data Points')
            ax[i].set(xlim = (np.min(times)-0.1, np.max(times)+0.1), ylim = (np.min(distances)-0.1,np.max(distances)+0.1), 
                                        ylabel = 'Displacement (m)', xlabel = 'Time (s)')

            chi2_object = Chi2Regression(quad_fit, times, distances, distances_err)
            chi2_object.errordef = 1
            fit = Minuit.Minuit(chi2_object, a = 2, v0 = 1, s0 = 1)
            fit.migrad()
        
            ## Use fit to calculate total uncertainty on s, taking into account the uncertainty on t
            time_err = 1e-4 # The sampling rate is 5 kHz
            derivatives = (quad_fit(times + 5e-4, *fit.values[:]) - quad_fit(times - 5e-4, *fit.values[:])) / 1e-3
  
            distances_total_err = np.sqrt(distances_err ** 2 + derivatives ** 2 * time_err ** 2  )
            
            # repeat fit with new uncertainties
            chi2_object = Chi2Regression(quad_fit, times, distances, distances_total_err)
            chi2_object.errordef = 1
            fit = Minuit.Minuit(chi2_object, a = 2, v0 = 1, s0 = 1)
            fit.migrad()
            x_vals = np.linspace(np.min(times), np.max(times), 1000)
            fit_vals =  quad_fit(x_vals, *fit.values[:])

            ax[i].plot(x_vals, fit_vals, 'k-', linewidth = 2)

            acc = fit.values['a']
            acc_err = fit.errors['a']
            a_vals[i] = acc
            a_errors[i] = acc_err

            Ndof = len(distances) - 3
            chi2 = fit.fval
            prop = stats.chi2.sf(chi2, Ndof)

            d = {'Fit:': '0.5 * a *t^2 + v0*t + s0' ,'a': [fit.values['a'],fit.errors['a']], 'v0': [fit.values['v0'],fit.errors['v0']], 's0': [fit.values['s0'],fit.errors['s0']],
                'Ndof': Ndof, 'Chi2': chi2, 'Prop': prop}

            text = nice_string_output(d, extra_spacing=2, decimals=3)
            add_text_to_ax(0.05, 0.9, text, ax[i], fontsize=10)

        # Calculate average and chi2 value
        chi2_val = chi2_func(a_vals, a_errors)
        points = 4

     
      
        fig_listf[j].tight_layout()

    #Calculate dTheta for big and small ball and convert back to degrees
    dTheta_big = calc_dtheta(np.array([a_BB_norm]), np.array([a_BB_rev]), theta) * 180 / np.pi
    dTheta_small = calc_dtheta(np.array([a_SB_norm]), np.array([a_SB_rev]), theta) * 180 / np.pi

    err_dTheta_big = calc_error_dtheta(np.array([a_BB_norm]), np.array([a_BB_rev]), theta, np.array([aerr_BB_norm]), np.array([aerr_BB_rev]), theta_err ) * 180/ np.pi
    err_dTheta_small = calc_error_dtheta(np.array([a_SB_norm]), np.array([a_SB_rev]), theta, np.array([aerr_SB_norm]), np.array([aerr_SB_rev]), theta_err ) * 180 / np.pi

    chi2big = np.sum((dTheta_big - dTheta_big.mean())**2 / err_dTheta_big ** 2 )
    chi2small = np.sum((dTheta_small - dTheta_small.mean())**2 / err_dTheta_small ** 2 )


    dTheta_tot = np.array([dTheta_big, dTheta_small]).flatten()
    err_dTheta_tot = np.array([err_dTheta_big, err_dTheta_small]).flatten()

    chi2tot = np.sum((dTheta_tot - np.average(dTheta_tot)) ** 2 / (err_dTheta_tot) ** 2)


    plt.show()


if __name__ == '__main__':
    main()

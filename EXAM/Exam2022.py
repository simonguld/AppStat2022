# Author: Simon Guldager Andersen
# Date (latest update): 20/1-2023

### SETUP --------------------------------------------------------------------------------------------------------------------

## Imports:
import os, sys
import iminuit
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib import rcParams
from scipy import stats, integrate, optimize, constants

sys.path.append('Appstat2022\\External_Functions')
from ExternalFunctions import Chi2Regression, BinnedLH, UnbinnedLH
from ExternalFunctions import nice_string_output, add_text_to_ax    # Useful functions to print fit results on figure


## Change directory to current one
os.chdir('AppStat2022\\EXAM')


### FUNCTIONS ----------------------------------------------------------------------------------------------------------------

def generate_dictionary(fitting_object, Ndatapoints, chi2_fit = True, chi2_suffix = None, subtract_1dof_for_binning = False):

    Nparameters = len(fitting_object.values[:])
    if chi2_suffix is None:
        dictionary = {'Entries': Ndatapoints}
    else:
        dictionary = {f'({chi2_suffix}) Entries': Ndatapoints}


    for i in range(Nparameters):
        dict_new = {f'{fitting_object.parameters[i]}': [fitting_object.values[i], fitting_object.errors[i]]}
        dictionary.update(dict_new)
    if subtract_1dof_for_binning:
        Ndof = Ndatapoints - Nparameters - 1
    else:
        Ndof = Ndatapoints - Nparameters
    if chi2_suffix is None:
        dictionary.update({f'Ndof': Ndof})
    else:
        dictionary.update({f'({chi2_suffix}) Ndof': Ndof})

    if chi2_fit:
        chi2 = fitting_object.fval
        p = stats.chi2.sf(chi2, Ndof)
        if chi2_suffix is None:
            dictionary.update({'Chi2': chi2, 'Prop': p})
        else:
            dictionary.update({f'({chi2_suffix}) Chi2': chi2, f'({chi2_suffix}) Prop': p})
    return dictionary

def calc_weighted_mean(x, dx):
    """
    returns: weighted mean, error on mean, Ndof, Chi2, p_val
    """
    assert(len(x) > 1)
    assert(len(x) == len(dx))
    
    var = 1 / np.sum(1 / dx ** 2)
    mean = np.sum(x / dx ** 2) * var

    # Calculate statistics
    Ndof = len(x) - 1
    chi2 = np.sum((x - mean) ** 2 / dx ** 2)
    p_val = stats.chi2.sf(chi2, Ndof)

    return mean, np.sqrt(var), Ndof, chi2, p_val

def calc_mean_std_sem(x, ddof = 1):
    """ returns mean, std, sem (standard error on mean)
    """
    Npoints = len(x)
    mean = x.mean()
    std = x.std(ddof = ddof)
    sem = std / np.sqrt(Npoints)
    return mean, std, sem

def calc_cov_matrix(data, ddof = 1):
    """assuming that each column represents a separate variable"""
    rows, cols = data.shape
    cov_matrix = np.empty([cols, cols])

    for i in range(cols):
        for j in range(i, cols):
            if ddof == 0:
                cov_matrix[i,j] = np.mean(data[:,i] * data[:,j]) - data[:,i].mean() * data[:,j].mean()
                cov_matrix[j,i] = cov_matrix[i,j]
            elif ddof == 1:
                cov_matrix[i,j] = 1/(rows - 1) * np.sum((data[:,i] - data[:,i].mean())*(data[:,j] - data[:,j].mean()))
                cov_matrix[j,i] = cov_matrix[i,j]
            else:
                print("The degrees of freedom must be 0 or 1")
                return None

    return cov_matrix

def calc_corr_matrix(x):
    """assuming that each column of x represents a separate variable"""
   
    data = x.astype('float')
    rows, cols = data.shape
    corr_matrix = np.empty([cols, cols])
 
    for i in range(cols):
        for j in range(i, cols):
                corr_matrix[i,j] = (np.mean(data[:,i] * data[:,j]) - data[:,i].mean() * data[:,j].mean()) / (data[:,i].std(ddof = 0) * data[:,j].std(ddof = 0))

        corr_matrix[j,i] = corr_matrix[i,j]
    return corr_matrix

def prop_err(dzdx, dzdy, x, y, dx, dy, correlation = 0):
    """ derivatives must takes arguments (x,y)
    """
    var_from_x = dzdx(x,y) ** 2 * dx ** 2
    var_from_y = dzdy (x, y) ** 2 * dy ** 2
    interaction = 2 * correlation * dzdx(x, y) * dzdy (x, y) * dx * dy

    prop_err = np.sqrt(var_from_x + var_from_y + interaction)

    if correlation == 0:
        return prop_err, np.sqrt(var_from_x), np.sqrt(var_from_y)
    else:
        return prop_err

def prop_err_3var(dzdx, dzdy, dzdt, x, y, t, dx, dy, dt):
    """ derivatives must takes arguments (x,y,t). Asummes no correlation between x,y,t
    """
    var_from_x = dzdx(x,y,t) ** 2 * dx ** 2
    var_from_y = dzdy (x, y, t) ** 2 * dy ** 2
    var_from_t = dzdt(x, y, t) ** 2 * dt ** 2

    prop_err = np.sqrt(var_from_x + var_from_y + var_from_t)

    return prop_err, np.sqrt(var_from_x), np.sqrt(var_from_y), np.sqrt(var_from_t)

def rejection_sampling_uniform(function, fmax, bounds, Npoints, verbose = True):

    ## ALGORITHM: The goal is to x1, ...., xn points distributed according to f on domain D
    # 1) generate a point x in D distributed according to a probability distribution g/area(g) (so g can be any curve...) enclosing f, ie f <= g on D
    # 2) generate uniformly a random point u in [0,g(x)]
    # 3) if u<f(x), keep x. 
    # 4) Rinse and repeat until desired number of points has been aquired

    # generate values according to f using rejection samping
    r = np.random

    xmin, xmax = bounds[0], bounds[1] 

    ## Using rejection/accepting method with both a constant pdf as well as 1/(1+x)
    x_accepted = np.empty(0)
    N_try = int(3 * Npoints)
    N_accum = 0
  

    while x_accepted.size < Npoints:
        # Construct N_points points by accepting/rejecting using a uniform pdf
        ## First, we construct N_try points uniformly on [xmin,xmax]
        r_vals = xmin + (xmax - xmin) * r.rand(N_try)
        ## Next, we construct another set of uniform random values in [0,fmax = y]
        #u_vals = fmax * r.rand(N_try)
        u_vals = r.uniform(0, fmax, size = N_try)
        ## Finally, we keep only the r_vals values satisfying u_vals < f(r_vals)
        mask = (u_vals < function(r_vals))
        vals = function(r_vals)

        x_accepted = np.r_['0', x_accepted, r_vals[mask]]

        # store total number of calculated samples
        N_accum += N_try
        # update N_try
        N_try = int(3 * (Npoints - x_accepted.size))

        if x_accepted.size > Npoints:
            x_accepted = x_accepted[:Npoints]

    eff_uni = Npoints / N_accum
    eff_err = np.sqrt(eff_uni*(1-eff_uni) / N_accum)
 
    if verbose:
        print("efficiency uniform: ", f'{eff_uni:6.3f}', "\u00B1 ", f'{eff_err:6.3f}')

    return x_accepted, eff_uni, eff_err

def get_statistics_from_fit(fitting_object, Ndatapoints, subtract_1dof_for_binning = False):
    
    Nparameters = len(fitting_object.values[:])
    if subtract_1dof_for_binning:
        Ndof = Ndatapoints - Nparameters - 1
    else:
        Ndof = Ndatapoints - Nparameters
    chi2 = fitting_object.fval
    prop = stats.chi2.sf(chi2, Ndof)
    return Ndof, chi2, prop

def do_chi2_fit(fit_function, x, y, dy, parameter_guesses, verbose = True):

    chi2_object = Chi2Regression(fit_function, x, y, dy)
    fit = iminuit.Minuit(chi2_object, *parameter_guesses)
    fit.errordef = iminuit.Minuit.LEAST_SQUARES

    if verbose:
        print(fit.migrad())
    else:
        fit.migrad()
    return fit

def do_LH_fit(fit_func, x_vals, paramters_guesses, bound, unbinned = True, bins = None, extended = True, verbose = True):
    if unbinned:
        LH_object = UnbinnedLH(fit_func, x_vals, bound = bound, extended = extended)
    else:
        LH_object = BinnedLH(fit_func, x_vals, bound = bound, bins = bins, extended = extended)
    
    fit = iminuit.Minuit(LH_object, *paramters_guesses)
    fit.errordef = iminuit.Minuit.LIKELIHOOD
    
    if verbose:
        print(fit.migrad())
    else:
        fit.migrad()
    return fit

def evaluate_likelihood_fit (fit_function, fmax, parameter_val_arr, log_likelihood_val, bounds, Ndatapoints, \
     Nsimulations, Nbins = 0, extended = True, unbinned = True):
    """
    fit_function is assumed to have the form f(x, *parameters), with x taking values in bounds
    Returns:: LL_values, p_value
    """
    LL_values = np.empty(Nsimulations)
    Nsucceses = 0
    max_iterations = 2 * Nsimulations
    iterations = 0
   
     # Simulate data
    while Nsucceses < Nsimulations and iterations < max_iterations:
        iterations += 1
   
        # Create values distributed according to fit_function

        x_vals, _, _ = rejection_sampling_uniform(lambda x: fit_function(x, *parameter_val_arr), fmax, bounds = bounds, Npoints = Ndatapoints, verbose = False)

        if 0:
            plt.figure()
            Nbins = 40
            plt.hist(x_vals, bins = Nbins, range = bounds)
            def gaussian_binned(x, N, mean, std):
                bin_width = (bounds[1] - bounds[0]) /  Nbins
                return N * bin_width * 1 / (np.sqrt(2 * np.pi) * std) * np.exp(-0.5 * (x-mean) ** 2 / std ** 2)

            xx = np.linspace(bounds[0], bounds[1], 500)

            plt.show()
          

        # Construct fitting object
        if unbinned:
            LLH_object = UnbinnedLH(fit_function, x_vals, bound = (bounds[0], bounds[1]), extended = extended)
            fit = iminuit.Minuit(LLH_object, *parameter_val_arr)
        else:
            LLH_object =  BinnedLH(fit_function, x_vals, bins = Nbins, bound = (bounds[0], bounds[1]), extended = extended)
            fit = iminuit.Minuit(LLH_object, *parameter_val_arr)

        fit.errordef = iminuit.Minuit.LIKELIHOOD
        fit.migrad()
        print(fit.fval)
        
        if 0:
            print("sim data points : ", len(x_vals))
            print("fit params: ", fit.values[:])
            plt.figure()
            Nbins = 50
            plt.hist(x_vals, bins = Nbins, range = bounds)
            def func_binned(x):
                bin_width = (bounds[1] - bounds[0]) /  Nbins
                return Ndatapoints * bin_width * fit_function(x, *fit.values[:])

            xx = np.linspace(bounds[0], bounds[1], 500)
            plt.plot(xx, func_binned(xx), 'r-')
            plt.show()

        if fit.fmin.is_valid:
            LL_values[Nsucceses] = fit.fval
            Nsucceses += 1
        else:
            print(f"ERROR: Fit did not converge for simulation no. {Nsucceses}. Log likelihood value is not collected.")

    mask = (LL_values > log_likelihood_val)
    p_value = len(LL_values[mask]) / Nsimulations

    return LL_values, p_value

def plot_likelihood_fits(LL_values, p_val, log_likelihood_val):
        Nsimulations = len(LL_values)
        fig0, ax0 = plt.subplots(figsize = (6,4))
        ax0.set_xlabel('Log likelihood value', fontsize = 18)
        ax0.set_ylabel('Count', fontsize = 18)
        ax0.set_title('Simulated log-likehood values', fontsize = 18)

        LL_std = LL_values.std(ddof = 1)
        counts, edges, _ = plt.hist(LL_values, bins = int(Nsimulations / 10), histtype = 'step', lw = 2, color = 'red');
        x_vals = 0.5 * (edges[:-1] + edges[1:])
        ax0.set_ylim(0,np.max(counts+5))
        ax0.plot([log_likelihood_val, log_likelihood_val], [0,np.max(counts)], 'k--', label = 'Log likelihood value (from fit)', lw = 2)

        ax00 = ax0.twinx()
        ax00.set_yticks(np.arange(0,1.1, 0.1))
        print("counts ",counts.sum())
        val_cumsum = np.cumsum(counts) / counts.sum()

        ax00.plot(x_vals, val_cumsum, 'k-', label = 'Cumulative distribution', lw = 2)
        # Adding fit results to plot:
        d = {'Entries':   Nsimulations,
            'Prob':     p_val}

        text = nice_string_output(d, extra_spacing=2, decimals=3)
        add_text_to_ax(0.05, 0.75, text, ax0, fontsize=16)

        fig0.legend( fontsize = 16, bbox_to_anchor = (0.25,0.65,0.25,0.25))
        fig0.tight_layout()
        return None

def one_sample_test(sample_array, exp_value, error_on_mean = None, one_sided = False, small_statistics = False):
    """ Assuming that the errors to be used are the standard error on the mean as calculated by the sample std 
    Returns test-statistic, p_val
    If a scalar sample is passed, the error on the mean must be passed as well, and large statistics is assumed
    """
    if np.size(sample_array) == 1:
        assert(error_on_mean is not None)
        assert(np.size(error_on_mean) == 1)
        assert(small_statistics == False)
        SEM = error_on_mean
        x = sample_array
    else:
        x = sample_array.astype('float')
        Npoints = np.size(x)
        SEM = x.std(ddof = 1) / np.sqrt(Npoints)
    
    test_statistic = (np.mean(x) - exp_value) / SEM

    if small_statistics:
        p_val = stats.t.sf(np.abs(test_statistic), df = Npoints - 1)
    else:
        p_val = stats.norm.sf(np.abs(test_statistic))

    if one_sided:
        return test_statistic, p_val
    else:
        return test_statistic, 2 * p_val

def two_sample_test(x, y, x_err = None, y_err = None, one_sided = False, small_statistics = False):
    """
    x,y must be 1d arrays of the same length. 
    If x and y are scalars, the errors on the means x_rr and y_rr must be passed as well, and small_statistics must be False
    If x and y are arrays, the standard errors on the mean will be used to perform the test

    Returns: test_statistics, p_val
    """
    Npoints = np.size(x)
    assert(np.size(x) == np.size(y))

    if x_err == None:
        SEM_x = x.std(ddof = 1) / np.sqrt(Npoints)
    else:
        assert(small_statistics == False)
        assert(np.size(x_err) == 1)
        SEM_x = x_err
        
    if y_err == None:
        SEM_y = y.std(ddof = 1) / np.sqrt(Npoints)
    else:
        assert(small_statistics == False)
        assert(np.size(y_err) == 1)
        SEM_y = y_err
        

    test_statistic = (np.mean(x) - np.mean(y)) / (np.sqrt(SEM_x ** 2 + SEM_y ** 2)) 

    if small_statistics:
        p_val = stats.t.sf(np.abs(test_statistic), df = 2 * (Npoints - 1))
    else:
        p_val = stats.norm.sf(np.abs(test_statistic))
    if one_sided:
        return test_statistic, p_val
    else:
        return test_statistic, 2 * p_val

def runstest(residuals):
   
    N = len(residuals)

    indices_above = np.argwhere(residuals > 0.0).flatten()
    N_above = len(indices_above)
    N_below = N - N_above

    print(N_above)
    print("bel", N_below)
    # calculate no. of runs
    runs = 1
    for i in range(1, len(residuals)):
        if np.sign(residuals[i]) != np.sign(residuals[i-1]):
            runs += 1

    # calculate expected number of runs assuming the two samples are drawn from the same distribution
    runs_expected = 1 + 2 * N_above * N_below / N
    runs_expected_err = np.sqrt((2 * N_above * N_below) * (2 * N_above * N_below - N) / (N ** 2 * (N-1)))

    # calc test statistic
    test_statistic = (runs - runs_expected) / runs_expected_err

    print("Expected runs and std: ", runs_expected, " ", runs_expected_err)
    print("Actual no. of runs: ", runs)
    # use t or z depending on sample size (2 sided so x2)
    if N < 50:
        p_val = 2 * stats.t.sf(np.abs(test_statistic), df = N - 2)
    else:
        p_val = 2 * stats.norm.sf(np.abs(test_statistic))

    return test_statistic, p_val

def calc_fisher_discrimminant(data1, data2, weight_normalization = 1):
    data_1 = data1.astype('float')
    data_2 = data2.astype('float')

    means_1 = np.sum(data_1, axis = 0)
    means_2 = np.sum(data_2, axis = 0)

    cov_1 = calc_cov_matrix(data_1, ddof = 1)
    cov_2 = calc_cov_matrix(data_2, ddof = 1)


    covmat_comb_inv = np.linalg.inv(cov_1 + cov_2)  
    weights = covmat_comb_inv @ (means_1 - means_2) / weight_normalization

    fisher_discrimminant_1 = np.sum((weights) * data_1, axis = 1) 
    fisher_discrimminant_2 = np.sum((weights) * data_2, axis = 1)
 
    return fisher_discrimminant_1, fisher_discrimminant_2, weights

def calc_ROC(hist1, hist2, signal_is_to_the_right_of_noise = True, input_is_hist = True, bins = None, range = None) :
    """
    This function is a modified version of code written by Troels Petersen
    Calculate ROC curve from two histograms (hist1 is signal, hist2 is background):
    if input_is_hist = False, the input entries are assume to be arrays
    returns: False positive rate, True positive rate
    """
    # First we extract the entries (y values) and the edges of the histograms:
    # Note how the "_" is simply used for the rest of what e.g. "hist1" returns (not really of our interest)
    if input_is_hist:
        y_sig, x_sig_edges, _ = hist1 
        y_bkg, x_bkg_edges, _ = hist2
    else:
        y_sig, x_sig_edges = np.histogram(hist1, bins = bins, range = range)
        y_bkg, x_bkg_edges = np.histogram(hist2, bins = bins, range = range)
    

    # Check that the two histograms have the same x edges:
    if np.array_equal(x_sig_edges, x_bkg_edges) :
        
        # Extract the center positions (x values) of the bins (both signal or background works - equal binning)
        x_centers = 0.5*(x_sig_edges[1:] + x_sig_edges[:-1])
        
        # Calculate the integral (sum) of the signal and background:
        integral_sig = y_sig.sum()
        integral_bkg = y_bkg.sum()

        # Initialize empty arrays for the True Positive Rate (TPR) and the False Positive Rate (FPR):
        TPR = np.zeros_like(y_sig).astype('float') # True positive rate (sensitivity)
        FPR = np.zeros_like(y_sig).astype('float') # False positive rate ()
        
        # Loop over all bins (x_centers) of the histograms and calculate TN, FP, FN, TP, FPR, and TPR for each bin:
        for i, x in enumerate(x_centers): 
   
            # The cut mask
            cut = (x_centers < x)
            
            # True positive
            TP = np.sum(y_sig[~cut]) / integral_sig    # True positives
            FN = np.sum(y_sig[cut]) / integral_sig     # False negatives
        
            if TP == 0 and FN == 0:
                TPR[i] = 0
            else:
                TPR[i] = TP / (TP + FN)                    # True positive rate
          
            
            # True negative
            TN = np.sum(y_bkg[cut]) / integral_bkg      # True negatives (background)
            FP = np.sum(y_bkg[~cut]) / integral_bkg     # False positives
            if TN == 0 and FP == 0:
                FPR[i] = 0
            else:
                FPR[i] = FP / (FP + TN)                     # False positive rate   
         
        if signal_is_to_the_right_of_noise:
            return FPR, TPR
        else:
            # If hist2 is signal, TPR is actually FPR and the other way around
            return TPR, FPR
    
    else:
        AssertionError("Signal and Background histograms have different bins and/or ranges")

def calc_sample_purity(hist_signal, hist_background, numpy_hist = False, signal_is_to_the_right_of_noise = True) :
    """
    Big thanks to Troels for generously providing the code upon which this function is based
    """
    # First we extract the entries (y values) and the edges of the histograms:
    # Note how the "_" is simply used for the rest of what e.g. "hist1" returns (not really of our interest)
    if numpy_hist:
        y_sig, x_sig_edges = hist_signal
        y_bkg, x_bkg_edges = hist_background
    else:
        y_sig, x_sig_edges, _ = hist_signal
        y_bkg, x_bkg_edges, _ = hist_background

    if signal_is_to_the_right_of_noise is False:
        x_sig_edges *= - 1
        x_bkg_edges *= - 1
    
    # Check that the two histograms have the same x edges:
    if np.array_equal(x_sig_edges, x_bkg_edges) :
        
        # Extract the center positions (x values) of the bins (both signal or background works - equal binning)
        x_centers = 0.5*(x_sig_edges[1:] + x_sig_edges[:-1])
        
        # Calculate the integral (sum) of the signal and background:
        integral_sig = y_sig.sum()
        integral_bkg = y_bkg.sum()
    
        # Initialize empty arrays for the sample purity: TP/(TP+FP)
        SP = np.zeros_like(y_sig) # True positive rate (sensitivity)
        
        # Loop over all bins (x_centers) of the histograms and calculate TN, FP, FN, TP, FPR, and TPR for each bin:
        for i, x in enumerate(x_centers): 
            
            # The cut mask
            cut = (x_centers < x)
            
            # True positive
            sig_area = np.sum(y_sig[~cut])   # True positives
            bkg_area = np.sum(y_bkg[~cut])     # False positives
            if sig_area == 0 and bkg_area == 0:
                SP[i] = 0
            else:
                SP[i] = sig_area / (sig_area + bkg_area)                    # False positive rate     

        if signal_is_to_the_right_of_noise is False:
            x_centers *= -1       
        return x_centers, SP
    else:
        AssertionError("Signal and Background histograms have different bins and/or ranges")


### MAIN ---------------------------------------------------------------------------------------------------------------------

def P1_1():
    #### consider scores of tests A and B both Gaussian with mean = 50, sigma  = 20

    ### what fraction will get a score in test A in range [55,66]?

    mean = 50
    std = 20

    cut1 = 55
    cut2 = 65

    # One 1 side
    p_further_than_cut1 = stats.norm.sf(cut1, loc = mean, scale = std)
    p_further_than_cut2 = stats.norm.sf(cut2, loc = mean, scale = std)
    p_between = p_further_than_cut1 - p_further_than_cut2


    print("P(55 < x < 65) = ", p_between)


    ### What uncertainty on the mean score do you obtain from 120 B test scores?
    N = 120
    print("uncertainty on mean from 120 B test scores: ", std / np.sqrt(N))


    ### If scores correlate with rho = 0.6, what fraction should get a score above 60 in both tests?
    rho = 0.6

    ## we will simulate Nsamples Nsimulations times from at bivariate normal distribution with mean and covariance matrix
    mean_vec = mean * np.ones(2)
    cov_matrix = np.array([[std ** 2, rho * std * std], [rho * std * std, std ** 2]])


    Nsamples = 10_000_0000.
    Nsimulations = 20

    fractions = np.empty(20)
   
   
    for i in range(Nsimulations):
        x = stats.multivariate_normal.rvs(mean = mean_vec, cov = cov_matrix, size = Nsamples)
        mask = ((x[:,0] > 60) & (x[:, 1] > 60))
        Nabove = np.size(x[mask][:,0])
        fractions[i] = Nabove / Nsamples

    f_mean, f_std, f_sem = calc_mean_std_sem(fractions)

    print("fraction above 60 and estimated error on mean: ", f_mean, "\u00B1", f_sem)

def P1_2():
    # game is designed s.t. 40% of  people will win on average

    # if 20 random people play, what is the chance that 11 or more will win???

    p = 0.4
    n = 20
    k_crit = 11

    prop = stats.binom.sf(k = k_crit - 1, n = n, p = p)

    print("P(k>=11,n=20,p=0.4) = ", prop)

def P1_3():
    pass

def P2_1():
           ## Initialize variables
    x, dx = 1.033, 0.014
    y, dy = 0.07, 0.23

    # z1 = xyexp(-y)
    z1 = lambda x,y: x * y * np.exp(-y)
    dz1dx = lambda x, y:  y * np.exp(-y) 
    dz1dy = lambda x, y:   (1 - y) * x * np.exp(-y)

    # z2 = (y+1)^2 / (x-1)
    z2 = lambda x, y: (y + 1) ** 3 / (x - 1)
    dz2dx = lambda x,y: - (y + 1) ** 3 / (x - 1) ** 2
    dz2dy = lambda x, y: 3 * (y + 1) ** 2 / (x - 1)
 
    ## Find uncertainties of z1 and z2 if uncorrelated
    err_z1, err_z1_from_x, err_z1_from_y = prop_err(dz1dx, dz1dy, x, y, dx, dy)
    err_z2, err_z2_from_x, err_z2_from_y = prop_err(dz2dx, dz2dy, x, y, dx, dy)

    print("\nFor z1: ")
    print("z1 = ", z1(x,y))
    print(f'Total uncertainty: {err_z1},  error propagated from x: {err_z1_from_x},  error propagated from y: {err_z1_from_y}')
    print("\nFor z2: ")
    print("z1 = ", z2(x,y))
    print(f'Total uncertainty: {err_z2},  error propagated from x: {err_z2_from_x},  error propagated from y: {err_z2_from_y}')

    ## Find uncertainties if correlation coefficient = 0.95
    correlation = 0.4
    
    err_z1 = prop_err(dz1dx, dz1dy, x, y, dx, dy, correlation = correlation)
    err_z2 = prop_err(dz2dx, dz2dy, x, y, dx, dy, correlation = correlation)

    print(f'Error on z1 if correlation = {correlation}:  {err_z1}')
    print(f'Error on z2 if correlation = {correlation}:  {err_z2}') 

def P2_2():
    ## Measurement of a tumor's depth (cm) was done with two methods. First one gave 4 measurements with uncertainties, the other one 12 without
    data1 = np.array([5.50, 5.61, 4.88, 5.07, 5.26])
    data1_err = np.array([0.10, 0.21, 0.15, 0.14, 0.13])

    print("This is geophysics. We'll do significance level of 5%")

    ## Do the measurements with uncertainty agree? Do those without?

    # With unc:
    const_func = lambda x,a: a
    fit = do_chi2_fit(const_func, np.arange(len(data1)), data1, data1_err, np.array([5.3]))
    Ndof, chi2, p_val = get_statistics_from_fit(fit, len(data1))

    print("Doing chi2-fit with const function to check consistency. Ndof, chi2, p_val = ", Ndof, chi2, p_val)

    print("For data1 with outlier: mean = ", fit.values['a'], "\u00b1", fit.errors['a'])
    print("verific, ", calc_weighted_mean(data1, data1_err))

    dist_from_weighted_mean = (data1 - fit.values['a']) / data1_err

    print("distances from weighted mean in units of their std \n", dist_from_weighted_mean)


    ##Excluding outlier
    data_new = np.delete(data1, 2)
    data_new_err = np.delete(data1_err, 2)
    mean_new, sem_new, Ndof_new, chi2_new, p_new = calc_weighted_mean(data_new, data_new_err)

    print("mean, sem, ndof, chi2, p_val excluding the outlier: ", calc_weighted_mean(data_new, data_new_err))

    rho_exp = 5.514

    t,p = one_sample_test(mean_new, exp_value = rho_exp, error_on_mean = sem_new)

    print("2 tailed 1 sample t (t,p) = ", t, p )
    print("The measurements are inconsistent with the expected value")
    
    print(calc_weighted_mean(data_new - rho_exp, data_new_err))

    chi2 = np.sum((data_new - rho_exp) ** 2/data_new_err ** 2)
    print("calculating chi square and prop as cross check : ", chi2, stats.chi2.sf(chi2,4))

def P2_3():
             ## Initialize variables
    x, dx = 1.04, 0.27 ## semi major axis
    y, dy = 0.71, 0.12 ## eccentricity

    # z1 = pi x ** 2 * np.sqrt(1 - y ** 2)
    z1 = lambda x,y: np.pi * x ** 2 * np.sqrt(1 - y ** 2)
    dz1dx = lambda x, y:  2 * np.pi * x  * np.sqrt(1 - y ** 2)
    dz1dy = lambda x, y: - np.pi * x ** 2 * y * 1 / np.sqrt(1 - y ** 2)

    ## Find uncertainties of z1 and z2 if uncorrelated
    err_z1, err_z1_from_x, err_z1_from_y = prop_err(dz1dx, dz1dy, x, y, dx, dy)


    print("\nFor Area: ")
    print("Area = ", z1(x,y))
    print(f'Total uncertainty: {err_z1},  error propagated from x: {err_z1_from_x},  error propagated from y: {err_z1_from_y}')
    print("\nFor z2: ")


    # NOW REPEAT FOR LOWER CIRFUMFERENCE BOUNDARY

                ## Initialize variables
    x, dx = 1.04, 0.27 ## semi major axis
    y, dy = 0.71, 0.12 ## eccentricity

    # z1 = 4 * x * np.sqrt(2 - y ** 2)
    z1 = lambda x,y: 4 * x * np.sqrt(2 - y ** 2)
    dz1dx = lambda x, y:  4  * np.sqrt(2 - y ** 2)
    dz1dy = lambda x, y: - 4 * x * y * 1 / np.sqrt(2 - y ** 2)

    ## Find uncertainties of z1 and z2 if uncorrelated
    err_z1, err_z1_from_x, err_z1_from_y = prop_err(dz1dx, dz1dy, x, y, dx, dy)


    print("\nFor lower cirfumference bound: ")
    print("C_lower = ", z1(x,y))
    print(f'Total uncertainty: {err_z1},  error propagated from x: {err_z1_from_x},  error propagated from y: {err_z1_from_y}')
    print("\nFor z2: ")
   
    ## uper circumference bound c_upper = pi/2 * c_lower, so
    c_lower = z1(x,y)
    err_c_lower = err_z1
    c_upper = np.pi / 2 * z1(x,y)
    err_c_upper = np.pi / 2 * err_z1
    print("upper bound and Error on upper bound: ", c_upper, err_c_upper)
   
    #assuming that the mean of C is given by the mid-point with gaussian errors
    C = 0.5 * (c_upper + c_lower)
    dC = 0.5 * np.sqrt(err_c_upper ** 2 + err_c_lower ** 2)

    print("value of C and uncertainty: ", C, dC)

def P3_1():

    ## assume that both distributions are gaussian around the mean, which is shifted from each other with time_diff
    time_diff = 130
    ship_err = 50
    truck_err = 120
    time_err = np.sqrt(truck_err ** 2 + ship_err ** 3)
    minutes_in_day = 24 * 60


    N_simulation = 1_000_000
        
    truck_vals = stats.norm.rvs(scale = truck_err, size = N_simulation)
    ship_vals = stats.norm.rvs(loc = time_diff, scale = ship_err, size = N_simulation)
    diff_vals =  truck_vals - ship_vals


    fig, ax = plt.subplots()
    range = (np.min(truck_vals), max(np.max(ship_vals),np.max( truck_vals)))
    bins = 200


    counts_t, edges, _ = ax.hist(truck_vals, bins = bins, range = range,
     histtype = 'stepfilled', lw = 2, alpha = 0.5, label = 'Histogram of truck arrival times')
    

    counts_s, edges, _ = ax.hist(ship_vals, bins = bins, range = range,
     histtype = 'stepfilled', lw = 2, alpha = 0.5, label = 'Histogram of ship departure times')


    counts_diff, edges, _ = ax.hist(diff_vals, bins = bins, range = range,
    histtype = 'stepfilled', lw = 2, alpha = 0.5, label = 'Histogram of time differences')

    x_vals = 0.5 * (edges[:-1] + edges[1:])
    #errors = np.sqrt(counts)
    #mask = (counts > 0)
    #N_non_empty = len(mask)

    ax.set(xlabel = 'Time (minutes)', ylabel = 'Count', title = 'Simulated time distributions')
    #ax.errorbar(x_vals, counts, errors, fmt = 'k.', elinewidth=1, capsize=1, capthick=1)

    ## Calc fraction where trucks arrive later than ship departure
    fraction = np.sum(counts_diff[x_vals > 0])  / N_simulation
    mask_below = (x_vals < 0)
    mask_above = (x_vals > 0)

    x_sum_below = - np.sum(x_vals[mask_below] * counts_diff[mask_below])
    x_sum_above = np.sum((x_vals[mask_above]+minutes_in_day + diff_vals.mean()) * counts_diff[mask_above]) 
    x_sum = x_sum_above + x_sum_below
  
    print("waiting av ", np.sum(x_sum_below) / ((1-fraction)* N_simulation))
    print("waiting av ", np.sum(x_sum_above) / ((fraction)* N_simulation))
    print("waiting av ", np.sum(x_sum / (N_simulation)))
 
    sim, read = True, True
    N_simulation =  3 * N_simulation
    t_range = np.arange(231-50,231+50,1)
    if sim:
        ## CALC Waiting times:
       
        waiting_time = np.empty_like(t_range, dtype = 'float')

        for i, time in enumerate(t_range):
            truck_vals = stats.norm.rvs(scale = truck_err, size = N_simulation)
            ship_vals = stats.norm.rvs(loc = time, scale = ship_err, size = N_simulation)
            diff_vals =  truck_vals - ship_vals

            mask_above = (diff_vals > 0)
            mask_below = (diff_vals < 0 )
    

            sum_below = - np.sum(diff_vals[mask_below])
            penalty = minutes_in_day
            #penalty = diff_vals.mean() + minutes_in_day
            if len(mask_above) > 1:
                sum_above = np.sum(time - diff_vals[mask_above] + penalty)
            else:
                sum_above = 0

          #  print(sum_below/sum_above)

            Nabove = len(diff_vals[mask_above])

            fraction= Nabove / N_simulation

        # print(sum_above/(fraction * N_simulation), sum_below/((1-fraction) * N_simulation))

            mean_waiting_time = (sum_above + sum_below) / N_simulation

            waiting_time[i] = mean_waiting_time
        #np.savetxt('time_points.txt', waiting_time)
    if 0:
        if read:
            waiting_time = np.loadtxt('time_points.txt')

    fig2,ax2 = plt.subplots()

    ax2.plot(t_range, waiting_time, '.')
    ax2.set(xlabel = r"$\Delta t$ - Av. time between truck arrival and ship departure (min)", ylabel = 'Average waiting time (min)', \
        title = "Average waiting time for different arrival-departure differences")

    cutoff = 200
    fit_range = 20
    bottom, upper = 231 - fit_range, 231 + fit_range
    mask0 = np.argwhere((t_range > bottom) & (t_range < upper))
    t_cut = t_range[mask0]
    def fit_func(x,x0, a,c):
        #b = - 2 * a * x0
        return a * (x-x0) ** 2  + c

    def fit_func_sum(x0, a,c):
        return np.sum((waiting_time[mask0] - fit_func(t_cut,x0,a,c)) ** 2)


    fit = iminuit.Minuit(fit_func_sum, x0 = 223, a = 1, c = 270)
    fit.errordef = 1
    print(fit.migrad())

 

       # Plot figure text
    d = {"Entries": len(t_cut), "a": [fit.values['a'],fit.errors['a']], "x0": [fit.values['x0'],fit.errors['x0']], "c": [fit.values['c'],fit.errors['c']]}
    text = nice_string_output(d, extra_spacing=0, decimals=5)
    add_text_to_ax(0.10, 0.95, text, ax2, fontsize=13)


    ax2.plot(t_cut, fit_func(t_cut, *fit.values[:]), label = r'Fit: $y = a(x-x_0)^2+c$')
    ax2.legend()
    plt.show()

    Ntrials = 20
    fraction_arr = np.empty(Ntrials)



    for i in np.arange(Ntrials):
        truck_vals = stats.norm.rvs(scale = truck_err, size = N_simulation)
        ship_vals = stats.norm.rvs(loc = time_diff, scale = ship_err, size = N_simulation)
        diff_vals =  truck_vals - ship_vals


        mask_above = (diff_vals > 0)
        mask_below = (diff_vals < 0 )

        Nabove = len(diff_vals[mask])

        fraction_arr[i] = Nabove / N_simulation
        print(fraction_arr[i])
    mean, std, sem = calc_mean_std_sem(fraction_arr)
    print("Using 20 samples: Fraction of trucks that have to wait till next day, mean, std, sem  ", mean, std, sem)


    fig.tight_layout()
    ax.legend(loc = 'upper left')
    plt.show()

def P3_2():
    ## Generate 1000 values according to f(x) = 1/sigma^2 exp(-0.5 x^2 /sigma ^2) for sigma = 2. Plot. Fit. How well can you determine sigma?

      ### Fit distribution with Gaussian
    Npoints = 1000
    sigma = 2

    # By the transformation method, values are gen. according to f by (but r!=1)
    transformation = lambda r: np.sqrt( - 2 * sigma ** 2 * np.log(1 - r))

    # Gen 1000 uniform values. Subtract tuning to ensure that r != 1
    tuning = 1e-12
    vals_uniform = np.random.rand(Npoints) - tuning 
    y_vals = transformation(vals_uniform)


    fig, ax = plt.subplots()
    range = (0, 15)
    bins = 70

    def fit_func(x, sigma):
        bin_width = (range[1] - range[0]) /  bins
        return Npoints * bin_width * x / sigma ** 2 * np.exp(-0.5 * (x) ** 2 / sigma ** 2)


    def fit_func_LH(x, sigma):
        return Npoints *  x / sigma ** 2 * np.exp(-0.5 * (x) ** 2 / sigma ** 2)


    counts, edges, _ = ax.hist(y_vals, bins = bins, range = range,
     histtype = 'stepfilled', lw = 2, alpha = 0.5, label = 'Simulated values')
    

    x_vals = 0.5 * (edges[:-1] + edges[1:])
    errors = np.sqrt(counts)
    mask = (counts > 0)
    N_non_empty = len(counts[mask])

    ax.set(xlabel = 'x value', ylabel = 'Count', title = 'Values simulated according to f')
    ax.errorbar(x_vals, counts, errors, fmt = 'k.', elinewidth=1, capsize=1, capthick=1)

   
    parameter_guesses = np.array([2])
    fit_chi2 = do_chi2_fit(fit_func, x_vals[mask], counts[mask], errors[mask], parameter_guesses)
    x_fit = np.linspace(range[0], range[1], 1000)

    if 1:
        fit_LH = do_LH_fit(fit_func_LH, y_vals , fit_chi2.values, bound = range, unbinned = True, extended = True)
        if 0:
            fmax = fit_func_LH(fit_LH.values['mean'], *fit_LH.values)
            Nsimulations = 100
            LL_values, p_val = evaluate_likelihood_fit(fit_func_LH, fmax = fmax, parameter_val_arr = fit_LH.values, \
                log_likelihood_val = fit_LH.fval, bounds = range, Ndatapoints = len(x_vals), Nsimulations = Nsimulations)

            plot_likelihood_fits(LL_values, p_val, log_likelihood_val = fit_LH.fval, Nsimulations = Nsimulations)
        
        fit_vals_LH = fit_func(x_fit, *fit_LH.values[:])
        ax.plot(x_fit, fit_vals_LH, label = 'Unbinned LH fit')
        
    
    fit_vals_chi2 = fit_func(x_fit, *fit_chi2.values[:])
    ax.plot(x_fit, fit_vals_chi2, label = r'Fit: $f(x) = \frac{x}{\sigma^2} \exp(-\frac{x^2}{2 \sigma^2})$')
    
    d = {'Entries': Npoints, 'Bins': N_non_empty}
    d0 = generate_dictionary(fit_chi2, Ndatapoints = N_non_empty, chi2_suffix='chi2')
    d.update(d0)
    del d['(chi2) Entries']
    d.update({'(LH) sigma': [fit_LH.values['sigma'],fit_LH.errors['sigma']]})
    text = nice_string_output(d, extra_spacing=0, decimals=2)
    add_text_to_ax(0.5, 0.7, text, ax, fontsize=14)

    bins_with_enough_statistics = len(counts[counts > 4]) / N_non_empty

    print("Fraction of bins with enough statistics: ", bins_with_enough_statistics)

    ax.legend()
    fig.tight_layout()
    plt.show(block = False)


    ## Test the 1/sqrt(N) scaling of uncertainty of sigma for N in [50,5000]

    Nrange = np.arange(1,100_000, 10)

    sigmas = np.empty_like(Nrange, dtype = 'float')
    sigmas_err = np.empty_like(sigmas, dtype = 'float')



    for i, N in enumerate(Nrange):
        vals_uniform = np.random.rand(N) - tuning 
        y_vals = transformation(vals_uniform)


        def fit_func(x, sigma):
            bin_width = (range[1] - range[0]) /  bins
            return N * bin_width * x / sigma ** 2 * np.exp(-0.5 * (x) ** 2 / sigma ** 2)

        counts, edges = np.histogram(y_vals, bins = bins, range = range)
     
        x_vals = 0.5 * (edges[:-1] + edges[1:])
        errors = np.sqrt(counts)
        mask = (counts > 0)
        N_non_empty = len(counts[mask])

   
        parameter_guesses = np.array([2])
        fit_chi2 = do_chi2_fit(fit_func, x_vals[mask], counts[mask], errors[mask], parameter_guesses, verbose = False)
    
        if not fit_chi2.fmin.is_valid:
            print(f'Fit failed for N = {i}')
        sigmas[i] = fit_chi2.values['sigma']
        sigmas_err[i] = fit_chi2.errors['sigma']

            ## Load data x is time in month, y is monthly income in M$, dy is error on income.

    ### STEP 1: Plot the thing
    fig0, ax0 = plt.subplots()

   




    def misfit(N, sigma0):
        return np.sum(np.abs(sigmas_err - sigma0 / np.sqrt(N)))

    def fit_func_free(N, sigma1, a):
        return sigma1 / N ** a

    def fit_func(N, sigma0):

        return sigma0 / N ** 0.5


    uncertainties = sigmas_err / 10
    parameter_guesses = np.array([1.4, 0.5])
    fit_free = do_chi2_fit(fit_func_free, Nrange, sigmas_err, uncertainties, parameter_guesses)

   # fit_free = iminuit.Minuit(lambda sigma0, a: misfit(Nrange, sigma0, a), *parameter_guesses)
    fit_free.errordef = 1
    fit_free.migrad()

    
   
    fit_post = do_chi2_fit(fit_func, Nrange, sigmas_err, uncertainties, np.array([1]))

   # ax0.errorbar(Nrange, sigmas_err, uncertainties, fmt = 'k.', elinewidth=.5, capsize=.5, capthick=.5)

    ax0.plot(Nrange, sigmas_err, '.', lw = 2, alpha = 0.5)
 
    ax0.plot(Nrange, fit_func(Nrange, *fit_post.values[:]), label = r"Fit 1: $\frac{\sigma_0}{\sqrt{N}}$")
    ax0.plot(Nrange, fit_func_free(Nrange, *fit_free.values[:]), label = r"Fit 2: $\frac{\sigma_1}{N^a}$")
    ax0.set_xlim((Nrange[0]-20, Nrange[-1]))
    ax0.set_ylim((0,0.4))

    ax0.set(xlabel = 'N - No. of simulated values', ylabel = r'Uncertainty on $\sigma$', title = r"Uncertainty on $\sigma$ against no. of sim. values")
   

    d = {"Entries": len(Nrange)}
    d0 = generate_dictionary(fit_post, Ndatapoints = len(Nrange), chi2_suffix='fit 1')
    d.update(d0)
    d0 = generate_dictionary(fit_free, Ndatapoints = len(Nrange), chi2_suffix='fit 2')
    d.update(d0)
    del d['(fit 1) Entries']
    del d['(fit 2) Entries']
    # Plot figure text
    text = nice_string_output(d, extra_spacing=0, decimals=2)
    add_text_to_ax(0.05, 0.9, text, ax0, fontsize=13)

    ax0.legend()
    fig0.tight_layout()
    plt.show()

def P4_1():
      ## Load data, 1st column records health label (0 = healthy, 1 = ill), last 3 columns report concentrations of substances A,B,C in blood
    temp, blood, age, labels = np.loadtxt('data_AnorocDisease.csv', usecols = (1,2,3,4), skiprows = 1, delimiter = ',', unpack = True)
    assert (labels.shape == blood.shape == age.shape == temp.shape)

    #patients 0-800 are known to be sick or healthy. 801: unknown
    cutoff_known = 800
    N_known = len(temp[:801])

    indices_ill = np.argwhere(labels == 1).flatten()
    indices_healthy = np.argwhere(labels == 0).flatten()
    indices_unknown = np.argwhere(labels == -1).flatten()

    temp_ill = temp[indices_ill]
    blood_ill = blood[indices_ill]
    age_ill = age[indices_ill]

    temp_healthy = temp[indices_healthy]
    blood_healthy = blood[indices_healthy]
    age_healthy = age[indices_healthy]

    data_ill = np.block([[temp_ill],[blood_ill],[age_ill]]).T
    data_healthy = np.block([[temp_healthy],[blood_healthy],[age_healthy]]).T

    data_unknown = np.block([[temp[indices_unknown]],[blood[indices_unknown]],[age[indices_unknown]]]).T

    plt.scatter(temp_ill, blood_ill, label = 'ii ill')
    plt.scatter(temp_healthy, blood_healthy, label = 'healthy')
    plt.legend()
    plt.show()

    ## Find the best separation between healthy and ill people using any variable of combination
    fig2,ax2 = plt.subplots(ncols = 3, figsize = (14,7))
    ax2.flatten()
    fig4, ax4 = plt.subplots()
    bins = (100, 100, 100)

    list_healthy = [temp_healthy, blood_healthy, age_healthy]
    list_ill = [temp_ill, blood_ill, age_ill]
    list_names = ['Temperature', 'Blood pressure', 'Age']


    for i, (healthy, ill) in enumerate(zip(list_healthy, list_ill)):
        hist_healthy = ax2[i].hist(healthy, bins = bins[i], histtype = 'stepfilled', alpha = .3, lw = 2, label = f'{list_names[i]} - Healthy')
        hist_ill = ax2[i].hist(ill, bins = bins[i], histtype = 'step', alpha = .6, lw = 2, label = f'{list_names[i]} - Ill')


        ax2[i].legend()
        fig2.supylabel("Count", fontsize = 18)
        fig2.suptitle("Sample distribution for each feature", fontsize = 20)
        ax2[i].set(xlabel = f'{list_names[i]}')


        range = (min(np.min(healthy), np.min(ill)), max(np.max(healthy), np.max(ill)))
      
        if healthy.mean() > ill.mean():
            signal = 'Healthy'
            FPR, TPR = calc_ROC(healthy, ill , input_is_hist = False, bins = bins[i], range = range)
        else:
            signal = 'Ill'
            FPR, TPR = calc_ROC(ill, healthy , input_is_hist = False, bins = bins[i], range = range)
       

        ax4.plot(FPR, TPR, label = f'{list_names[i]}: Signal = {signal}')


  
    ax4.set(xlabel = r'FPR = $\beta$', ylabel = r'TPR = 1 - $\alpha$', title = 'ROC curves for each feature', xlim = (-0.1,1))


    fig4.tight_layout()
    fig2.tight_layout()
  
   
    plt.show(block = False)


    ## TEST if age distribution is the same between healthy and sick peoples

    print("Testing whether age dist. for healthy and sick people are the same with 2-sided Smirnov:")
    
    res_smirnov = stats.ks_2samp(age_ill, age_healthy)
    print("Smirnov test statistic and p value: ", res_smirnov[0], res_smirnov[1])
    print(res_smirnov)


    ## Separate the two groups as well as possible, and use this to estimate the number of infected people in the unkown group.
    # plot fisher in ROC curve and also do a histogram
    print("Combining all variables in a fisher discrimminant yields")
    fisher_ill, fisher_healthy, weights = calc_fisher_discrimminant(data_ill, data_healthy, weight_normalization =  1000)

    fig5, ax5 = plt.subplots()
    bins = 50
    range_fisher = (min(np.min(fisher_healthy), np.min(fisher_ill)), max(np.max(fisher_healthy), np.max(fisher_ill)))
    hist_fisher_ill = ax5.hist(fisher_ill, bins = bins, range = range_fisher, histtype = 'step', alpha = .6, lw = 2, label = 'Fisher distribution - Ill')
    hist_fisher_healthy = ax5.hist(fisher_healthy, bins = bins, range = range_fisher, histtype = 'stepfilled', alpha = .2, lw = 2, label = 'Fisher distribution- Healthy')

    ax5.set(xlabel = 'Fisher discrimminant', ylabel = 'Count', title = 'Fisher separation of healthy and ill people')
    FPR_fisher, TPR_fisher = calc_ROC(hist_fisher_ill, hist_fisher_healthy)
    ax4.plot(FPR_fisher, TPR_fisher, label = f'Fisher: Signal = Ill')
    alpha = 0.15

    critical_indices = np.argwhere(np.abs(TPR_fisher - (1-alpha)) < 0.02).flatten()
    critical_index = np.max(critical_indices)
    _, edges, _ = hist_fisher_ill
    x_vals = 0.5 * (edges[:-1] + edges[1:])
    x_val_cutoff = x_vals[critical_index] 

    ax5.plot([x_val_cutoff, x_val_cutoff], [0,18], 'r--', label = r'Cutoff for $\alpha = 0.15$')
    print("fisher cutoff value =", x_val_cutoff)
    print("using alpha = ", 1- TPR_fisher[critical_index], "implies beta = ", FPR_fisher[critical_index], " and a concentration cutoff of ", x_vals[critical_index])

    ax4.plot([-0.1,FPR_fisher[critical_index]], [TPR_fisher[critical_index], TPR_fisher[critical_index]], 'k--', label = f'alpha = {1-TPR_fisher[critical_index]:.2f}')
    ax4.plot([FPR_fisher[critical_index],FPR_fisher[critical_index]], [0, TPR_fisher[critical_index]], 'k--', label = f'beta = {FPR_fisher[critical_index]:.2f}')
   
 

    ##Transform unkown data via fisher weights
    #fisher_discrimminant_1 = np.sum((weights) * data_1, axis = 1) 
    range = (np.min(data_unknown[:,0]), np.max(data_unknown[:,0]))
  
    fisher_unknown = np.sum(weights * data_unknown, axis = 1)
    

    fig6, ax6 = plt.subplots()
    range = (np.min(fisher_unknown), np.max(fisher_unknown))
    counts, _,_ = ax6.hist(fisher_unknown, bins = bins, range = range, histtype='step',  alpha = .6, lw = 2, label = 'Fisher distribution - Unknown')
    ax6.set(xlabel = 'Fisher discrimminant', ylabel = 'Count', title = 'Fisher separation of the unkown test group')
    ax6.plot([x_val_cutoff, x_val_cutoff], [0,15], 'k--', label = r'Cutoff for $\alpha = 0.15$')

    number_ill_est = counts[x_vals > x_val_cutoff].sum()

    print("From the Fisher separation we estimate the no. of ill people ", number_ill_est)

    print("Fisher weights: ", weights)


    ## If a new patient as T = 38.6, and prior prop of being sick is p = 0.01, what is the prop that the patient is ill=?
    p = 0.01
    T_cutoff = 38.6
    range = (min(np.min(temp_healthy), np.min(temp_ill)), max(np.max(temp_healthy), np.max(temp_ill)))
    _, edges = np.histogram(temp_healthy, bins = bins, range = range)
    t_vals = 0.5 * (edges[1:] + edges[:-1])
    T_bin_cutoff = np.min(np.argwhere(np.abs(t_vals - T_cutoff) < 0.07).flatten())
    FPR, TPR = calc_ROC(temp_ill, temp_healthy , input_is_hist = False, bins = bins, range = range)
    ax4.plot([FPR[T_bin_cutoff], FPR[T_bin_cutoff]], [0, TPR[T_bin_cutoff]], 'r--', label = f'beta = {FPR[T_bin_cutoff]:.2f}')
    ax4.plot([-0.1, FPR[T_bin_cutoff]], [TPR[T_bin_cutoff], TPR[T_bin_cutoff]], 'r--', label = f'alpha = {1-TPR[T_bin_cutoff]:.2f}')
    print(T_bin_cutoff)


    ax4.legend()
    ax5.legend()
    ax6.legend()
    fig5.tight_layout()
    plt.show()

def P4_2():
    data = pd.read_csv("data_CountryScores.csv", sep = ',')
  
    for col in data.columns:
        assert(len(data['Country']) == len(data[col]))

    Nentries = len(data)
    print(data.columns)
    
    GDP = pd.Series.to_numpy(data[:]['GDP'], dtype = 'float')
    PopSize = pd.Series.to_numpy(data[:]['PopSize'], dtype = 'float')
    happiness = pd.Series.to_numpy(data[:]['Happiness-index'], dtype = 'float')
    education = pd.Series.to_numpy(data[:]['Education-index'], dtype = 'float')

    index_DK = 30

    ### Determine mean, median 25 and 75 quantiles of the GDP
    GDP_mean = GDP.mean()
    GDP_median = np.median(GDP)
    GDP_1st_quantile = np.quantile(GDP, 0.25)
    GDP_3rd_quantile = np.quantile(GDP, 0.75)

    print("GDP mean median first and third quantile: ", GDP_mean, GDP_median, GDP_1st_quantile, GDP_3rd_quantile)


    ### Does log_10(PopSize) follow a Gaussian distribution?
    logPop = np.log10(PopSize)
    fig, ax = plt.subplots()
 
    Nbins = 25
    range = (np.min(logPop), np.max(logPop))
   

    ax.hist(logPop, bins = Nbins, range = range, histtype = 'stepfilled', alpha = 0.4, lw = 2, label = r'Histogram of $\log_{10}$(population size)')
    ax.set(xlabel = r'$\log_{10}(PopSize)$', ylabel = 'Count', title = 'Population size distribution')
   

    ## DO anderson and Fisher test
  
    res_anderson = stats.anderson(logPop, dist = 'norm')

    print("Applying the Anderson test to see if the distribution of log_10(PopSize) follows a normal distribution results in:")
    print("test statistic:", res_anderson.statistic)
    print("critical values of test statistics:", res_anderson.critical_values)
    print("corresponding significance levels: ", res_anderson.significance_level)
    print("since the test statistics is less than the critical value at significant 15 %, even at 15% the null hypothesis \
        that the distribution follows a normal distribution cannot be rejected.")

    print("Testing the same thing with 2-sided Smirnov (with mean and std equal to distribution):")
    
    res_smirnov = stats.kstest(logPop, lambda x: stats.norm.cdf(x, loc = logPop.mean(), scale = logPop.std(ddof = 1)))
    print("Smirnov test statistic and p value: ", res_smirnov[0], res_smirnov[1])
    print(res_smirnov)

    ax.legend()
    fig.tight_layout()
    plt.show(block = False)


    ## What are the Pearson and Spearman correlations between happiness and education indices?
    fig2, ax2 = plt.subplots()


    arr = np.block([[happiness],[education]]).T
    corr_arr = calc_corr_matrix(arr)
    spearmann_corr, _ = stats.spearmanr(happiness, education) #p_val for the hypohtesis that they are uncorrelated
    print("Pearson corr. between happines and education is ", corr_arr[0,1])
    print("Spearman corr. (and p-val) between happines and education is ", spearmann_corr)


    ax2.scatter(education, happiness, alpha = 0.5)
    ax2.plot([],[], ' ', label = r"$\rho_P$ = 0.76")
    ax2.plot([],[], ' ', label = r"$\rho_S$ = 0.80")
    ax2.set(xlabel = 'Education index', ylabel = 'Happiness index', title = "Relationship between happiness and education index")

    fig2.tight_layout()
    ax2.legend()
  


    ### Plot the happiness index against GDP and fit the relation. From this fit, what would you estimate the uncertainty to be on the happiness index?? 


    ### STEP 1: Plot the thing
    fig0, ax0 = plt.subplots(figsize = (12,8))

    ## RANK THEM
    ranking = np.argsort(GDP)
    GDP_ranked = GDP[ranking]
    happiness_ranked = happiness[ranking]
   # ax0.errorbar(x, y, dy, fmt = 'k.', elinewidth=1.5, capsize=1.5, capthick=1)
    ax0.plot(GDP_ranked, happiness_ranked, '.', alpha = 0.4, markersize = 7)
    ax0.set(xlabel = 'GDP', ylabel = 'Happiness index',  title = "Happiness index as a function of GDP")
    ax0.plot([],[], " ", label = r'Fit: $y = a \ln [b(x - 260)] + e$')
    offset = np.min(GDP)-1
    if 0:
        def power_func(x,a,b, c, d, e):
            return a * (np.abs(x - offset)) ** b + c * np.log (d *(x - offset))  + e

        def power_func_sum(a, b, c, d, e):
            return np.sum(np.abs(happiness_ranked - power_func(GDP_ranked,a,b, c, d, e)))

    if 1:
        def power_func(x,a,b,e): #, c, d, e):
            return  a * np.log(b * (x - offset)) + e #+ c * np.log (d *(x - offset))  + e  a * (np.abs(x - offset)) ** b

        def power_func_sum(a, b, e): #, c, d, e):
            return np.sum(np.abs(happiness_ranked - power_func(GDP_ranked,a,b, e)))

    param_guess = np.array([8.5, 0.0001, 8000]) # 1000, 0.00001, 8000])
   

    power_fit = iminuit.Minuit(power_func_sum, *param_guess)
    power_fit.errordef = 1.0
    print(power_fit.migrad())
    x_vals = np.linspace(GDP_ranked[0], GDP_ranked[-1], 500)
    y_vals = power_func(x_vals, *power_fit.values[:])
    ax0.plot(x_vals, y_vals, label = r"Fit (no uncertainty)")
    
    ## calc residuals and typical uncertainty
    residuals = happiness_ranked - power_func(GDP_ranked, *power_fit.values[:])
    std = residuals.std(ddof = 1) 
    print("typical happiness-index uncertainty: ", std)
    ax0.errorbar(GDP_ranked, happiness_ranked, std, fmt = 'k.', elinewidth=.7, capsize=.7, capthick=.7)

    err_fit = do_chi2_fit(power_func, GDP_ranked, happiness_ranked, std, power_fit.values)
    ax0.plot(x_vals, power_func(x_vals, *err_fit.values), label = 'Fit (with uncertainty)')
    d = generate_dictionary(err_fit, Ndatapoints = len(GDP_ranked))
    # Plot figure text
    text = nice_string_output(d, extra_spacing=0, decimals=5)
    add_text_to_ax(0.27, 0.53, text, ax0, fontsize=13)

    ### PLOT residuals

    ##runs 
    print("Residuals runs test", runstest(residuals), (len(np.argwhere(residuals > 0).flatten())))

    ax00 = ax0.twinx()
    shift = 12
    bottom = -16
    ax00.set(ylabel = 'Residuals in units of typical uncertainty')


    ax00.set_yticks(np.arange(bottom,bottom + 6,1), np.arange(-3,3,1))
    ax00.set_ylim((bottom,8))


    ax00.plot([GDP_ranked[0], GDP_ranked[-1]], [-shift,-shift], 'k-', linewidth = 2)
    ax00.plot([GDP_ranked[0], GDP_ranked[-1]], [1-shift, 1-shift], 'k--', linewidth = 1)
    ax00.plot([GDP_ranked[0], GDP_ranked[-1]], [-1-shift, -1-shift], 'k--', linewidth = 1)
    ax00.errorbar(GDP_ranked, (residuals)/std - shift, 1, fmt='ro', ecolor='k', elinewidth=.5, capsize=.6, capthick=.6, markersize = 2, label = 'Residuals')


    res_smirnov = stats.kstest(residuals, lambda x: stats.norm.cdf(x, loc = 0, scale = residuals.std(ddof = 1)))
    print("Reiduals Smirnov test statistic and p value: ", res_smirnov[0], res_smirnov[1])
    print(res_smirnov)

    res_anderson = stats.anderson(residuals, dist = 'norm')

    print("Applying the Anderson test to see if the distribution of residuals follows a normal distribution results in:")
    print("test statistic:", res_anderson.statistic)
    print("critical values of test statistics:", res_anderson.critical_values)
    print("corresponding significance levels: ", res_anderson.significance_level)

   

    legend = plt.legend()
    legend.get_frame().set_facecolor('none')
    ax0.legend(bbox_to_anchor = [0.62, 0.7])
    fig0.tight_layout()
    plt.show()


    if 0:
        durations = pd.Series.to_numpy(data[1:]['DurationInSec'], dtype = 'float')
        coast = pd.Series.to_numpy(data[1:]['UScoast'], dtype = 'float')
        hour_in_day =  pd.Series.to_numpy(data[1:]['HourInDay'], dtype = 'float')
        day_in_year =  pd.Series.to_numpy(data[1:]['DayInYear'], dtype = 'float')
        day_of_week =  pd.Series.to_numpy(data[1:]['DayOfWeek'], dtype = 'float') # Monday = 0 etc

        # Plot the distribution of duration of observation, calc med and median
        west_coast_indices = np.argwhere(coast == 1.).flatten()
        east_coast_indices = np.argwhere(coast == 2.).flatten()

        print(len(west_coast_indices), len(east_coast_indices))
        print("Mean and median for duration dist: ", durations.mean(), np.median(durations))

        fig, ax = plt.subplots()
        fig2, ax2 = plt.subplots()
        Nbins = 50
        print(coast[np.argmax(durations)])
        range = (np.min(durations), np.max(durations) + 1000)
        range2 = (0, 2800)

        ax.hist(durations, bins = Nbins, range = range, histtype = 'stepfilled', alpha = 0.4, lw = 2, label = 'West, East and Non-coast durations')
        ax2.hist(durations[west_coast_indices], bins = Nbins, range = range2, histtype = 'stepfilled', alpha = 0.4, lw = 2, label ='West coast durations')
        ax2.hist(durations[east_coast_indices], bins = Nbins, range = range2, histtype = 'stepfilled', alpha = 0.4, lw = 2, label ='East coast durations')
        
        ax.set(xlabel = 'Sighting duration (s)', ylabel = 'Count', title = 'Histogram of sighting duration')
        ax2.set(xlabel = 'Sighting duration (s)', ylabel = 'Count', title = 'Histogram of sighting duration')


        ax.legend()
        ax2.legend()
        fig.tight_layout()
        fig2.tight_layout()



        ### Do these durations follows same dist. on east and west coast?
        # Do 2 sample (2 tailed) smirnov test.
        test_statistic, p_val = stats.ks_2samp(durations[west_coast_indices], durations[east_coast_indices])
        print("KS test statistics and p-val: ", test_statistic, p_val, "and so null-hypothesis that they follow same distribution cannot be rejected")


        #### What is the corr. between day in year and time of day of observation?
        # Scatterplot the two
        fig3, ax3 = plt.subplots()
        bins = (60, 60)

        data_matrix = np.block([[day_in_year],[day_of_week]]).T
        corr_matrix = calc_corr_matrix(data_matrix)
    
        print("corr matrix: \n", corr_matrix)
        print("The lin. correlation coefficient = ", corr_matrix[0,1], " indicates that they are not correlated, but the scatter plot revales that they are, \
            particularly when it comes to sightings at night, in which the correlation seem to be parabolic")

        ax3.scatter(day_in_year, hour_in_day, alpha = 0.4, s = 16, marker = '.')
        _,_,_,im = ax3.hist2d(day_in_year, hour_in_day, range = ((0,356),(0,24)), bins = bins, cmap = 'Blues')

        ax3.set(xlabel = 'Day of the year', ylabel = 'Hour in day (hours)')
        plt.colorbar(im)
        fig3.tight_layout()
    

        # considering only west coast, is the disbtribution of number of observations uniform over 7 week days? How about considering Mon-Thurs?
        # assuming Poisson errors on no. of sightings!
        fig4, ax4 = plt.subplots()

        day_of_week_westcoast = day_of_week[west_coast_indices]
        events_per_weekday = np.empty(7)
        for i in np.arange(7):
            events_per_weekday[i] = len(np.argwhere(day_of_week_westcoast == i).flatten())

        events_error = np.sqrt(events_per_weekday)
        weekdays = np.arange(7)
        ax4.plot(weekdays, events_per_weekday, label = 'Total no. of sightings per weekday')
        ax4.errorbar(weekdays, events_per_weekday, events_error, fmt = 'k.', elinewidth=1.5, capsize=1.5, capthick=1)
        ax4.set_xticks(weekdays, ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], fontsize = 9)
        ax4.set(xlabel = 'Weekday', ylabel = 'Total no. of sightings', title = 'No. of sightings on diff. weekdays')

        fit_func_const = lambda x, a: a

        parameter_guess = np.array([1860])
        fit = do_chi2_fit(fit_func_const, weekdays, events_per_weekday, events_error, parameter_guess)
        ax4.plot(weekdays, fit.values['a'] * np.ones_like(weekdays), label = 'Const fit = a')

        parameter_guess = np.array([1860])
        fit2 = do_chi2_fit(fit_func_const, weekdays[:4], events_per_weekday[:4], events_error[:4], parameter_guess)
        ax4.plot(weekdays[:4], fit2.values['a'] * np.ones_like(weekdays[:4]), label = 'Const fit = b')

        d = generate_dictionary(fit, len(weekdays), chi2_suffix='a')
        d2 = generate_dictionary(fit2, len(weekdays[:4]), chi2_suffix='b')
        d.update(d2)
        # Plot figure text    
        text = nice_string_output(d, extra_spacing=0, decimals=4)
        add_text_to_ax(0.05, 0.7, text, ax4, fontsize=13)

        ax4.legend()
        fig4.tight_layout()
        plt.show()

def P5_1():
           ## Load data x is time in month, y is monthly income in M$, dy is error on income.
    A, dA, V, dV = np.loadtxt('data_GlacierSizes.csv', skiprows = 1, delimiter = ",", unpack = True)
    assert (A.shape == dA.shape == V.shape == dV.shape)

    area_index_sorting = np.argsort(A)
    A = A[area_index_sorting]
    dA = dA[area_index_sorting]
    V = V[area_index_sorting]
    dV = dV[area_index_sorting]


    ### STEP 1: Plot the thing
    fig0, ax0 = plt.subplots()
    ax0.plot(A,V, '.', alpha = 0.3, markersize = 7)
    ax0.errorbar(A, V, dV, dA, fmt = 'k.', elinewidth=1., capsize=1., capthick=.7)
    ax0.set(xlabel = r'Area ($km^2$)', ylabel = r'Volume $km^3$', title = "Glacier volume as a function of area")
 
    ## Which of the two have largest uncertainties??
    mask = ((dV / V)  > (dA / A))
 

    print("Fraction of points with larger rel. vol. uncertainty than rel area uncertainty ", len(dV[mask]) / len(dV))


    ## Ignore area uncertainties and fit with the expected V ~ A^(3/2) relation

    
    def fit_exp(x,a):
        return a * x ** (3/2)

    def fit_exp_offset(x,b, c):
        return b * x ** (3/2) + c

    def fit_power_free(x,d, e, g):
        return d * x ** e + g



    func_list = [fit_exp, fit_exp_offset, fit_power_free]
    params = [np.array([0.025]), np.array([0.023, 0.0074]), np.array([0.023, 1.42, 0.025])]
    fit_vals = [None] * len(func_list)
    labels = [r"Fit 0: $V = a A^{3/2}$", r"Fit 1: $V = b A^{3/2} + c$", r"Fit 2: $V = d A^{e} + g$"]
    d = {"Entries": len(A)}
    for i, func in enumerate(func_list):
        fit = do_chi2_fit(func, A, V, dV, params[i])
        fit_vals[i] = fit.values
        ax0.plot(A, func(A, *fit.values[:]), label = labels[i])

        d0 = generate_dictionary(fit, Ndatapoints = len(A), chi2_suffix = f'Fit {i}')
        d.update(d0)
        del d[f'(Fit {i}) Entries']


    # Plot figure text
    text = nice_string_output(d, extra_spacing=0, decimals=3)
    add_text_to_ax(0.05, 0.95, text, ax0, fontsize=13)


    ## DO runs test for expected fit
    residuals_exp = fit_exp(A, *fit_vals[0]) - V
    print("Runs test for expected fit (z,p): ", runstest(residuals_exp))

    ## DO runs test for exp fit with offset
    residuals_offset = fit_exp_offset(A, *fit_vals[-2]) - V
    print("Runs test for expected fit with offset (z,p): ", runstest(residuals_offset))

    ## DO runs test for power fit
    residuals_power = fit_power_free(A, *fit_vals[-1]) - V
    print("Runs test power fit with free exponent and offset (z,p): ", runstest(residuals_power))


    ## IS power fit actually better than exp fit with offset?? Wilk's test!
    chi2_val = d['(Fit 1) Chi2'] - d['(Fit 2) Chi2']
    Ndof = d['(Fit 1) Ndof'] - d['(Fit 2) Ndof']


    ## REPEAT INCLUDING area uncertainties

    ### STEP 1: Plot the thing
    fig1, ax1 = plt.subplots()
    ax1.plot(A,V, '.', alpha = 0.3, markersize = 7)
    ax1.set(xlabel = r'Area ($km^2$)', ylabel = r'Volume $km^3$', title = "Glacier volume as a function of area (incl. area unc.)")
 

    func_list = [fit_exp, fit_exp_offset, fit_power_free]
    params = [np.array([0.025]), np.array([0.023, 0.0074]), np.array([0.023, 1.42, 0.025])]
    fit_vals_tot = [None] * len(func_list)
    fit_vals_err = [None] * len(func_list)
    labels = [r"Fit 0: $V = a A^{3/2}$", r"Fit 1: $V = b A^{3/2} + c$", r"Fit 2: $V = d A^{e} + g$"]
    d = {"Entries": len(A)}
    exponent = [1.5, 1.5, 1.423]
    for i, func in enumerate(func_list):
        deltaA = 5e-3
       # derivatives = (func(A + deltaA, *fit_vals[i]) - func(A - deltaA, *fit_vals[i])) / (2 * deltaA)
        derivatives = exponent[i] * fit_vals[i][0] * A ** (exponent[i] - 1)
        print("fit vals i,", fit_vals[i][0])
        total_err = np.sqrt(dV ** 2 + derivatives ** 2 * dA ** 2  )

        fit = do_chi2_fit(func, A, V, total_err, params[i])
        fit_vals_tot[i] = fit.values
        fit_vals_err[i] = fit.errors
        ax1.plot(A, func(A, *fit.values[:]), label = labels[i])

        d0 = generate_dictionary(fit, Ndatapoints = len(A), chi2_suffix = f'Fit {i}')
        d.update(d0)
        del d[f'(Fit {i}) Entries']

    ax1.errorbar(A, V, total_err, dA, fmt = 'k.', elinewidth=1., capsize=1., capthick=.7)
    # Plot figure text
    text = nice_string_output(d, extra_spacing=0, decimals=3)
    add_text_to_ax(0.05, 0.95, text, ax1, fontsize=13)


    ## DO runs test for expected fit
    residuals_exp = fit_exp(A, *fit_vals_tot[0]) - V
    print("Runs test for expected fit (z,p): ", runstest(residuals_exp))

    ## DO runs test for exp fit with offset
    residuals_offset = fit_exp_offset(A, *fit_vals_tot[-2]) - V
    print("Runs test for expected fit with offset (z,p): ", runstest(residuals_offset))

    ## DO runs test for power fit
    residuals_power = fit_power_free(A, *fit_vals_tot[-1]) - V
    print("Runs test power fit with free exponent and offset (z,p): ", runstest(residuals_power))


    ## IS power fit actually better than exp fit with offset?? Wilk's test!
    chi2_val = d['(Fit 1) Chi2'] - d['(Fit 2) Chi2']
    Ndof = d['(Fit 1) Ndof'] - d['(Fit 2) Ndof']
    print(chi2_val)
    print("Wilks test: prop that fit1 and fit2 describe the same distriubiton: ", stats.chi2.sf(chi2_val, Ndof))


    ## ESTIMATE V AND dV for glacier with area 0.5
    A_crit = 0.5
    dA_crit = 5e-2

    V_crit = fit_power_free(A_crit, *fit_vals_tot[-1])

    ## FIND dV

    d = fit_vals_tot[-1][0]
    e = fit_vals_tot[-1][1]
    g = fit_vals_tot[-1][2]
    dd = fit_vals_err[-1][0]
    de = fit_vals_err[-1][1]
    dg = fit_vals_err[-1][2]

    print(" the dds ", dd, de, dg)
    print(np.mean(dV[:20]))
    ## Write eq above as z = d * A ** e + g
    dzdd = lambda d,e,A: A ** e
    dzde = lambda d,e,A: d * A ** (e) * np.log(A)
    dzdA = lambda d,e,A:  d * e * A ** (e-1)            
 

    dV_tot, _, _, _= prop_err_3var(dzdd, dzde, dzdA, d, e, A_crit, dd, de, dA_crit)

    dV_tot = np.sqrt(dg**2 + dV_tot ** 2)

    print("estimate vol of glacier with area 0.5: ", V_crit, "\u00B1", dV_tot)

    ax1.legend(loc = 'lower right')
    fig1.tight_layout()
    ax0.legend(loc = 'lower right')
    fig0.tight_layout()
    plt.show()
 
def P5_2():
    mu, sig, N = 0, .4, 1000
    x_lin = np.linspace(-1,1,N)
    sample_values = stats.norm.rvs(loc = mu, scale = sig, size = N) * np.exp(- 1.5 * x_lin)

    bounds = [-1, 1]
    alpha_guess = .5
    beta_guess = .5
    
    
    def func(x, alpha, beta):
        return np.maximum(1e-10, 1 + alpha * x + beta * np.power(x, 2))

    norm_const = lambda lower_bound, upper_bound, alpha, beta: upper_bound - lower_bound \
        + alpha/2 * (upper_bound ** 2 - lower_bound ** 2) + beta/3 * (upper_bound ** 3 - lower_bound ** 3)
    
    func_norm = lambda x, alpha, beta: 1 / (norm_const(bounds[0], bounds[1], alpha, beta)) * func(x, alpha, beta)
    fmax = 3 * func_norm(0, alpha_guess, beta_guess)
    LH_unbinned_object = UnbinnedLH(func_norm, sample_values, bound = bounds, extended = False)
    fit = iminuit.Minuit(LH_unbinned_object, alpha = alpha_guess, beta = beta_guess)
    fit.errordef = iminuit.Minuit.LIKELIHOOD
    print(fit.migrad())
    print(fit.fval)
    print("fmax", np.max(func_norm(x_lin, *fit.values[:])), fmax)

    samp2 = rejection_sampling_uniform(lambda x: func_norm(x, *fit.values[:]), fmax, bounds = bounds, Npoints = N)[0]

    LH_unbinned_object = UnbinnedLH(func_norm, samp2, bound = bounds, extended = False)
    fit0 = iminuit.Minuit(LH_unbinned_object, alpha = fit.values['alpha'], beta = fit.values['beta'])
    fit0.errordef = iminuit.Minuit.LIKELIHOOD
    print(fit0.migrad())
    print(fit0.fval)

    if 0:
        fig,ax = plt.subplots()
        bins = 50
        binwidth = (bounds[1] - bounds[0]) / bins
        ax.hist(sample_values, range = bounds, bins = bins, histtype='stepfilled', alpha = .4)
        func_norm_scale = lambda x: N * binwidth * func_norm(x, *fit.values[:])
        x_range = np.linspace(bounds[0], bounds[1], 400)
        fit_vals = func_norm_scale(x_range)
        ax.plot(x_range,fit_vals)
        LL_vals, p_val = evaluate_likelihood_fit(func_norm, fmax, fit.values[:], fit.fval, bounds = bounds, Ndatapoints = N, Nsimulations = 200, extended = False)
        plot_likelihood_fits(LL_vals, p_val, fit.fval)
    
  
    plt.show()


# Set plotting style
sns.set_theme()
sns.set_style("darkgrid")
sns.set_context("paper") #Possible are paper, notebook, talk and poster
rcParams['lines.linewidth'] = 2 
rcParams['axes.titlesize'] =  18
rcParams['axes.labelsize'] =  18
rcParams['xtick.labelsize'] = 12
rcParams['ytick.labelsize'] = 12
rcParams['legend.fontsize'] = 15
rcParams['font.family'] = 'serif'
rcParams['figure.figsize'] = (9,6)
rcParams['axes.prop_cycle'] = cycler(color = ['teal', 'navy', 'coral', 'plum', 'purple', 'olivedrab',\
         'black', 'red', 'cyan', 'yellow', 'khaki','lightblue'])
np.set_printoptions(precision = 5, suppress=1e-10)

## Set parameters and which problems to run
p1_1, p1_2, p1_3  = False, False, False
p2_1, p2_2, p2_3, p3_1, p3_2 = False, False, False, False, False
p4_1, p4_2, p5_1, p5_2 =  False, False, False, True


def main():
    
    problem_numbers = [p1_1, p1_2, p1_3, p2_1, p2_2, p2_3, p3_1, p3_2, p4_1, p4_2, p5_1, p5_2]
    f_list = [P1_1, P1_2, P1_3, P2_1, P2_2, P2_3, P3_1, P3_2, P4_1, P4_2, P5_1, P5_2]
    names = ['p1_1', 'p1_2', 'p1_3', 'p2_1', 'p2_2', 'p2_3', 'p3_1', 'p3_2', 'p4_1', 'p4_2', 'p5_1', 'p5_2']

    for i, f in enumerate(f_list):
        if problem_numbers[i]:
            print(f'\nPROBLEM {names[i][1:]}:')
            f()
   

if __name__ == '__main__':
    main()


# Author: Simon Guldager Andersen
# Date (latest update): 

### SETUP -----------------------------------------------------------------------------------------------------------------------------------

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
os.chdir('AppStat2022\\EXAM_2018')


### FUNCTIONS -------------------------------------------------------------------------------------------------------------------------------

def generate_dictionary(fitting_object, Ndatapoints, chi2_fit = True, chi2_suffix = None):

    Nparameters = len(fitting_object.values[:])
    if chi2_suffix is None:
        dictionary = {'Entries': Ndatapoints}
    else:
        dictionary = {f'({chi2_suffix}) Entries': Ndatapoints}


    for i in range(Nparameters):
        dict_new = {f'{fitting_object.parameters[i]}': [fitting_object.values[i], fitting_object.errors[i]]}
        dictionary.update(dict_new)
   
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

def get_statistics_from_fit(fitting_object, Ndatapoints):

    Ndof = Ndatapoints - len(fitting_object.values[:])
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

def evaluate_likelihood_fit (fit_function, fmax, parameter_val_arr, log_likelihood_val, bounds, Ndatapoints, Nsimulations, Nbins = 0, unbinned = True):
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

            plt.plot(xx, gaussian_binned(xx, *parameter_val_arr), 'r-')
          

        # Construct fitting object
        if unbinned:
            LLH_object = UnbinnedLH(fit_function, x_vals, bound = (bounds[0], bounds[1]), extended = True)
            fit = iminuit.Minuit(LLH_object, *parameter_val_arr)
        else:
            LLH_object =  BinnedLH(fit_function, x_vals, bins = Nbins, bound = (bounds[0], bounds[1]), extended = True)
            fit = iminuit.Minuit(LLH_object, *parameter_val_arr)

        fit.errordef = iminuit.Minuit.LIKELIHOOD
        fit.migrad()

        if fit.fmin.is_valid:
            LL_values[Nsucceses] = fit.fval
            Nsucceses += 1
        else:
            print(f"ERROR: Fit did not converge for simulation no. {Nsucceses}. Log likelihood value is not collected.")

    mask = (LL_values > log_likelihood_val)
    p_value = len(LL_values[mask]) / Nsimulations

    return LL_values, p_value

def plot_likelihood_fits(LL_values, p_val, log_likelihood_val, Nsimulations):

        fig0, ax0 = plt.subplots(figsize = (6,4))
        ax0.set_xlabel('Log likelihood value', fontsize = 18)
        ax0.set_ylabel('Count', fontsize = 18)
        ax0.set_title('Simulated log-likehood values', fontsize = 18)

        counts, edges, _ = plt.hist(LL_values, bins = 50, histtype = 'step', lw = 2, color = 'red');
        x_vals = 0.5 * (edges[:-1] + edges[1:])
        ax0.set_ylim(0,np.max(counts+5))
        ax0.plot([log_likelihood_val, log_likelihood_val], [0,np.max(counts)], 'k--', label = 'Log likelihood value (from fit)', lw = 2)

        ax00 = ax0.twinx()
        ax00.set_yticks(np.arange(0,1.1, 0.1))
        val_cumsum = np.cumsum(counts) / counts.sum()

        ax00.plot(x_vals, val_cumsum, 'k-', label = 'Cumulative distribution', lw = 2)
        # Adding fit results to plot:
        d = {'Entries':   Nsimulations,
            'Prob':     p_val}

        text = nice_string_output(d, extra_spacing=2, decimals=3)
        add_text_to_ax(0.05, 0.75, text, ax0, fontsize=16)
        #ax00.legend(loc='best', fontsize=16); # could also be # loc = 'upper right' e.g.
        #ax00.legend(loc = 'best', fontsize = 16)
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
 
    return fisher_discrimminant_1, fisher_discrimminant_2

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


### MAIN ------------------------------------------------------------------------------------------------------------------------------------

def P1_1():
    ## Peter goes to casino and puts money on black with
    p = 18/37

    ### In 50 games, what are the chances he will win exactly 25 times? 26 or  more?
    n = 50
    k25 = 25

    p25 = stats.binom.pmf(k = k25, n = n, p = p)
    p25plus = stats.binom.sf(k = k25, n = n, p = p)

    print("P(k = 25, n = 50, p = 18/37) = ", p25)
    print("P(k >= 26, n = 50, p = 18/37) = ", p25plus)

    ### How many games must he play in order to be 95% sure to win at least 20 times?
    ## --> Find n s.t. P(k>=20,n,p) = 0.95
    p_crit = 0.95
    kmin = 20

    n = 40
    prop = stats.binom.sf(kmin - 1, n, p)
    while prop < 0.95:
        print("(n,p) = (", n, prop,")")
        n += 1
        prop = stats.binom.sf(kmin - 1, n, p)
    
    print("So the answer is (n,p) = (", n, prop,")")

def P1_2():
    # Find prob. of a Gaussian val to lie between 1.2sig and 2.5sig away from mean

    cut1 = 1.2
    cut2 = 2.5

    # One 1 side
    p_further_than_cut1 = stats.norm.sf(cut1)
    p_further_than_cut2 = stats.norm.sf(cut2)
    p_between = p_further_than_cut1 - p_further_than_cut2

    p_two_sides = 2 * p_between

    print("P(1.2sigma < |x| < 2.5sigma) = ", p_two_sides)

def P1_3():
    ## No. of serious mistakes are counted daily.

    ## Should follow poisson (independent events, probably a fairly const. mean rate [which might slowly change over course of sev. years though])
    ### ---> But sum of Poisson is Poisson

    ## days with more than 8 mistakes are called critical. If there were 22 critical days in a year, estimate av. number of daily mistakes
    ## --> find lambda s.t. P(k>=8,lambda) ~ 22/365.25
    k_crit = 9
    p = 22/365  ## +++ counting uncertainty??

    ## ASSUME COUNTING UNCERTAINTY
    dp = np.sqrt(p * (1 - p) / 365)
    mean = 4.4
    increment = 1e-5
    iterations = 0
    misfit = stats.poisson.sf(k_crit - 1, mean) - p
    ## WE CANNOT DETERMINE IT MORE CLOSE THAN dp
    while np.abs(misfit) > dp and iterations < 100_000:
        iterations +=1
        mean += increment
        misfit = stats.poisson.sf(k_crit - 1, mean) - p
    
    mean_lower = mean
    print("Counting uncertainty = ", dp)
    print("mean_lower, sf and p", mean, stats.poisson.sf(k_crit - 1, mean), p)
    print("To find uncertainty, we solve the same system from above this time")

    mean = 5.4
    misfit = stats.poisson.sf(k_crit - 1, mean) - p
    while np.abs(misfit) > dp and iterations < 100_000:
        iterations +=1
        mean -= increment
        misfit = stats.poisson.sf(k_crit - 1, mean) - p

    print("mean_upper, sf and p", mean, stats.poisson.sf(k_crit - 1, mean), p)
    print("Assuming gaussian errors (and that varation of p within 1 sig translates to variation of mean within 1 sig) we find that")

    mean_both = 0.5 * (mean_lower + mean)
    uncertainty = (mean - mean_lower) / 2
    print("mean = ", mean_both, "\u00B1", uncertainty)

def P2_1():
    ## Measurement of a tumor's depth (cm) was done with two methods. First one gave 4 measurements with uncertainties, the other one 12 without
    data1 = np.array([2.05, 2.61, 2.46, 2.48])
    data1_err = np.array([0.11, 0.10, 0.13, 0.12])

    data2 = np.array([2.69, 2.71, 2.56, 2.48, 2.34, 2.79, 2.54, 2.68, 2.69, 2.58, 2.66, 2.70])

    print("This is medicine. We'll do significance level of 5%")

    ## Do the measurements with uncertainty agree? Do those without?

    # With unc:
    const_func = lambda x,a: a
    fit = do_chi2_fit(const_func, np.arange(len(data1)), data1, data1_err, np.array([2.45]))
    Ndof, chi2, p_val = get_statistics_from_fit(fit, len(data1))

    print("Doing chi2-fit with const function to check consistency. Ndof, chi2, p_val = ", Ndof, chi2, p_val)


    dist_from_weighted_mean = (data1 - fit.values['a']) / data1_err

    print("distances from weighted mean in units of their std \n", dist_from_weighted_mean)

    print("2 sided t test of 2.05 against weighted mean yields", 2 * stats.t.sf(3.21, df = 3))
    print("This is just barely below the significance level, but this point is almost 4 std away from its nearest neighbor \
        , and also considering the measurements without uncertainties, I think it is justified to throw this point away")

    ##Excluding outlier
    new_var_data1 = 1/np.sum(1/data1_err[1:] ** 2)
    new_mean_data1 = np.sum(data1[1:] / data1_err[1:] ** 2) * new_var_data1
    chi2 = np.sum((data1[1:] - new_mean_data1) ** 2 / (data1_err[1:] ** 2))
    print("ndof, chi2, p_val excluding the outlier: ", 2, chi2, stats.chi2.sf(chi2, 2))
 
    ## WITHOUT uncertainty. Do they agree??? Fit it, using std as estimate for uncertainty
    mean, std, sem = calc_mean_std_sem(data2)
    data2_err = std * np.ones_like(data2)
    fit2 = do_chi2_fit(const_func, np.arange(len(data2)), data2, data2_err, np.array([2.45]))
    Ndof2, chi22, p_val2 = get_statistics_from_fit(fit, len(data1))

    print("Doing chi2-fit with const function to check consistency. Ndof, chi2, p_val = ", Ndof2, chi22, p_val2)
    print("mean std sem ", mean, std, sem)

    ## Which of the two methods are most precise?
    print("For data1 with outlier: mean = ", fit.values['a'], "\u00b1", fit.errors['a'])
    print("For data1 without outlier: mean = ", new_mean_data1, "\u00b1", np.sqrt(new_var_data1))
    print("For data2 : mean = ", fit2.values['a'], "\u00b1", fit2.errors['a'])

    t, p = two_sample_test(new_mean_data1, mean, np.sqrt(new_var_data1), sem)
    print("t-value and p-value for combining means: ", 2 * stats.t.sf(np.abs(t),df = len(data1[1:]) + len(data2) - 2))
    ## so far data2 gives the best prediction. Does it make sense to include the 3 points of data 1?

    ## APP1: Combine means
    var_comb = 1 / (1/new_var_data1 + 1 / sem ** 2)
    mean_comb = (new_mean_data1 / new_var_data1 + mean / sem ** 2) * var_comb

    print("combining the means (best estimate), ", mean_comb, "\u00B1", np.sqrt(var_comb))

def P2_2():

    # Initialize constants 
    h = 6.626e-34
    c = 299.7e6
    k = 1.381e-23

    ## Initialize variables
    x, dx = 5.5e3, 0.29e3 ## === T (K)
    y, dy = 0.566e15, 0.025e15 ## === frequency (Hz)

    # z1 = (2hy^3 / c^2) * 1 /(exp([hy] / kT) - 1)
    z1 = lambda x, y: (2 * h * y ** 3 / c ** 2) * 1 / (np.exp((h * y) / (k * x)) - 1)
    dz1dx = lambda x, y: (2 * h * y ** 3 / c ** 2) * 1 / (np.exp((h * y) / (k * x)) - 1) ** 2 * (h * y / (k * x ** 2)) * np.exp((h * y) / (k * x))
    dz1dy = lambda x, y: 2 * h / c ** 2 * (3 * y ** 2 - y ** 3 * np.exp((h * y) / (k * x)) / (np.exp((h * y) / (k * x)) - 1) * h / (k * x)) * 1 / (np.exp((h * y) / (k * x)) - 1)

    ## Find uncertainties of z1 and z2 if uncorrelated
    err_z1, err_z1_from_x, err_z1_from_y = prop_err(dz1dx, dz1dy, x, y, dx, dy)
 
    print("Spectral radiance = ", z1(x,y), "\u00B1", err_z1)

    print(f'Total uncertainty: {err_z1},  error propagated from x: {err_z1_from_x},  error propagated from y: {err_z1_from_y}')

    ## Find uncertainties if correlation coefficient = 0.95
    correlation = 0.87
    
    err_z1 = prop_err(dz1dx, dz1dy, x, y, dx, dy, correlation = correlation)
  
    print(f'Error on z1 if correlation = {correlation}:  {err_z1}')
   
def P3_1():
     ## Consider distribution f = C *  (1 - exp(-ax)) for x in [0,2] and a = 2

    ## Determine normalization constant numerically
    range = [0.005, 1]
    f = lambda x: x ** (-0.9)
    area, err_area = integrate.quad(f, range[0], range[1])
    print("area: ", area, "error on area: ", err_area)
    C = 1 / area
    dC = np.sqrt((- 1 / area ** 2 ) ** 2 * err_area ** 2 )
    print(f'Value of C: {C:.14},  Uncertainty on C: {dC:.6}')

    # Find max value of f
    def f_norm(x):
        if x < range[0] or x > range[1]:
            return 0
        else:
            return C * x ** (-0.9)

    f = lambda x: C * x ** (-0.9)
    fmax = f(range[0]) ## monotonically decreasing
    print("fmax on domain: ", fmax)

    ## IT CAN BE INVERTED, so we can use rejection sampling

    cumulative_distribution_inv = lambda r: (range[0] ** 0.1 + r / (10 * C)) ** 10

    ## PRODUCE 10.000 values according to f and plot
    Npoints = 50_000
    uniform_vals = np.random.rand(Npoints)

    sample_vals_transformation = cumulative_distribution_inv(uniform_vals)

    ## Cross check using rejection sampling
    sample_vals_rejection, _, _ = rejection_sampling_uniform(f, fmax, bounds = range, Npoints = Npoints, verbose = True)


    bins = 100
    fig, ax = plt.subplots(ncols = 2, figsize = (10,10))
    ax = ax.flatten()

    sample_vals = [sample_vals_transformation, sample_vals_rejection]
    name_list = ['Transformation method', 'Rejection sampling']

    for i, values in enumerate(sample_vals):
        counts, edges, _ = ax[i].hist(values, bins=bins, range = range, histtype='step', label='histogram', linewidth = 2)
        ax[i].set(xlim=(range[0]-0.05, range[1]+0.05))
        ax[i].set_title(f'Sampled using {name_list[i]}')
        ax[i].set_xlabel( xlabel="x value", fontsize = 18)
        ax[i].set_ylabel( ylabel="Counts", fontsize = 18)
        x_vals = 0.5 * ( edges[:-1] + edges[1:])
        y_vals = counts
        y_err = np.sqrt(counts)
        mask = (y_vals > 0)
        N_non_empty = len(mask)

        ax[i].errorbar(x_vals, counts, y_err, fmt = 'k.', elinewidth=1.5, capsize=1.5, capthick=1)
        # Fit
        def fit_func(x, a):
            bin_width = (range[1]- range[0]) / bins
            scaling = Npoints * bin_width
            return scaling * C * x ** a 

        chi2_object = Chi2Regression(fit_func, x_vals[mask], y_vals[mask], y_err[mask])
        chi2_object.errordef = 1
        fit = iminuit.Minuit(chi2_object, a = -0.1)
        fit.migrad()

        # plot fit
        x_range = np.linspace(range[0], range[1], 1000)

        fit_vals =  fit_func(x_range, *fit.values[:])

        ax[i].plot(x_range, fit_vals, linewidth = 2, label = 'Fit')
        ax[i].legend(loc = 'best')

        # Get statistics
        Ndof = len(y_vals[mask]) - len(fit.values[:])
        chi2 = fit.fval
        prop = stats.chi2.sf(chi2, Ndof)
        rel_error = np.abs((fit.values['a'] - 3)) / 3
        d = {"Entries": Npoints, "Fit function": 'y = C * x^a', "fit param a": [fit.values['a'],fit.errors['a']], "rel error on a": rel_error,
                "Ndof": Ndof, "Chi squared": chi2, "Prop": prop}

        # Plot figure text
        text = nice_string_output(d, extra_spacing=2, decimals=3)
        add_text_to_ax(0.05, 0.90, text, ax[i], fontsize=13)

    fig.tight_layout()
    plt.show(block = False)

    ## gen. 1000 values each consisting of sum of 50 random values from f. 
    N_tsamples = 1000
    t_values = np.empty(N_tsamples)
    for i in np.arange(N_tsamples):
        t_values[i] = sample_vals_transformation[50 * i: 50 + 50 * i].sum()
    
    mean, std, sem = calc_mean_std_sem(sample_vals_rejection)
    t_mean, t_std, t_sem = calc_mean_std_sem(t_values)

    # APPROACH 1: Calc expectedd values from samples [but NOT the same sample as the one t is calculated from]
    t_val_exp = 50 * mean
    t_err_exp = np.sqrt(50 * sem **2)
    ## APPROACH 2: Explot that we have x(r), and a analytical expression for the mean of a function of 1 variable
    x_mean_analytical, x_mean_analytical_err = integrate.quad(cumulative_distribution_inv, range[0], range[1])
    t_val_analytical = 50 * x_mean_analytical
    t_val_analytical_err = np.sqrt(50 * x_mean_analytical_err ** 2)

    print("analytical expected t value and uncertainty: ", t_val_analytical, t_val_analytical_err)
    print("Numerical expected t value and uncertainty: ", t_val_exp, t_err_exp)
    print("actual t value and uncertainty: ", t_mean, t_sem)

    test_statistic, p_val = one_sample_test(t_values, exp_value = t_val_analytical, one_sided = False, small_statistics = False)
    print("2 sided 1 sample t-test for whether actual t value matches analytical expectation: ", test_statistic, p_val)

    if 1:
        bins = 50
        range = [4,18]
        fig, ax = plt.subplots(ncols = 1, figsize = (10,8))
        #ax = ax.flatten()

        sample_vals = [t_values]
        name_list = ['Sampled t-values']

        for i, values in enumerate(sample_vals):
            counts, edges, _ = ax.hist(values, bins=bins, range = range, histtype='step', label='histogram', linewidth = 2)
            ax.set(xlim=(range[0]-0.05, range[1]+0.05))
            ax.set_title(f'Sampled using {name_list[i]}')
            ax.set_xlabel( xlabel="x value", fontsize = 18)
            ax.set_ylabel( ylabel="Counts", fontsize = 18)
            x_vals = 0.5 * ( edges[:-1] + edges[1:])
            y_vals = counts
            y_err = np.sqrt(counts)
            mask = (y_vals > 0)
            N_non_empty = len(mask)

            ax.errorbar(x_vals, counts, y_err, fmt = 'k.', elinewidth=1.5, capsize=1.5, capthick=1)
            # Fit
           
            def gaussian_binned(x, N, mean, std):
                bin_width = (range[1] - range[0]) /  bins
                return N * bin_width * 1 / (np.sqrt(2 * np.pi) * std) * np.exp(-0.5 * (x-mean) ** 2 / std ** 2)

            def gaussian_LH(x, N, mean, std):
                return  N * 1 / (np.sqrt(2 * np.pi) * std) * np.exp(-0.5 * (x-mean) ** 2 / std ** 2)


            chi2_object = Chi2Regression(gaussian_binned, x_vals[mask], y_vals[mask], y_err[mask])
            chi2_object.errordef = 1
            fit = iminuit.Minuit(chi2_object, N = N_tsamples, mean = t_mean, std = t_std)
            print(fit.migrad())

            LH_object = UnbinnedLH(gaussian_LH, values, bound = range, extended = True)
            LH_object.errordef = iminuit.Minuit.LIKELIHOOD
            fit_LH = iminuit.Minuit(LH_object, *fit.values[:])
            print(fit_LH.migrad())

            # plot fit
            x_range = np.linspace(range[0], range[1], 1000)

            fit_vals =  gaussian_binned(x_range, *fit.values[:])
            fit_vals_LH = gaussian_binned(x_range, *fit_LH.values[:])
            ax.plot(x_range, fit_vals, linewidth = 2, label = 'Chi2 Gaussian fit')
            ax.plot(x_range, fit_vals_LH, linewidth = 2, label = 'LH Gaussian fit')
            ax.legend(loc = 'best')

            # Get statistics
            Ndof = len(y_vals[mask]) - len(fit.values[:])
            chi2 = fit.fval
            prop = stats.chi2.sf(chi2, Ndof)

            d = {"Entries": N_tsamples, "(chi2) N": [fit.values['N'],fit.errors['N']],  "(chi2) mean": [fit.values['mean'],fit.errors['mean']], \
                "(chi2) std": [fit.values['std'],fit.errors['std']], \
                    "(LL) N": [fit_LH.values['N'],fit_LH.errors['N']],  "(LL) mean": [fit_LH.values['mean'],fit_LH.errors['mean']], \
                "(LL) std": [fit_LH.values['std'],fit_LH.errors['std']],
                "(chi2) Ndof": Ndof, "Chi squared": chi2, "(chi2)Prop": prop}

            # Plot figure text
            text = nice_string_output(d, extra_spacing=2, decimals=3)
            add_text_to_ax(0.03, 0.95, text, ax, fontsize=13)

        ## Perform 2sample tests to check whether chi^2 and LL parameter values are consistent
        N = len(fit.values)

        print("Using two tailed two sample test of means to test whether chi^2 parameters are compatible with LL parameters:")
        for i in np.arange(N):
            param_chi, err_chi = fit.values[i], fit.errors[i]
            param_LL, err_LL = fit_LH.values[i], fit_LH.errors[i]
            print(f"param val {i} chi and LL", param_chi, param_LL)
            z, p_val = two_sample_test(param_chi, param_LL, err_chi, err_LL, one_sided = False, small_statistics = False)
            print("z-value and p value", z, p_val)

        
        fig.tight_layout()
        plt.show()

def P3_2():
    x = stats.norm.rvs(loc = -0.3, size = 10000)
    y = stats.norm.rvs(loc = 1., size = 8000)

    histx = plt.hist(x, bins = 150, range = (-4,4), label = 'x')
    histy = plt.hist(y, bins = 150, range = (-4,4))
    centers, SP = calc_sample_purity(histy,histx, signal_is_to_the_right_of_noise=True)
    plt.legend()
    plt.figure()
    plt.plot(centers, SP, label = 'signal = y')

    plt.legend()
    plt.show()

def P4_1():
    data = pd.read_csv("data_UfoSightings.txt",  skiprows=3, sep = '\t')
  
    for col in data.columns:
        assert(len(data['Date']) == len(data[col]))

    Nentries = len(data)
    print(data.columns)

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

def P4_2():
    frequency = np.array([185, 1149, 3265, 5475, 6114, 5194, 3067, 1331, 403, 105, 14, 4, 0], dtype = 'float')
    range = np.arange(13)
    mask = (frequency > 0)
    Ntrials = 12
    Nbins = 13
    N_non_empty = len(frequency[mask])
    p_naive = 1/3
    Ndatapoints = frequency.sum()
    fraction = frequency / Ndatapoints

    if 1:
        # ASSUMING BINOM. counting uncertainties. Ie the error on f = n/N is sqrt(f(1-f)/N)
        errors = np.sqrt(fraction * (1 - fraction) / Ndatapoints)
    if 0:
        # Assuming poisson errors on counts
        errors = np.sqrt(fraction)

    ## What distribution should the no. of 5s and 6s follows?
    # === Binomial. Although prob. reasonably well-described by a poisson given N = 13 and all p's relatively small.
    fig0, ax0 = plt.subplots()
    ax0.bar(range, fraction, alpha = 0.4)
    ax0.errorbar(range, fraction, errors, fmt = 'k.', elinewidth=1.5, capsize=1.5, capthick=1)

    ax0.set(xlabel = "Total no. of 5s and 6s", ylabel = 'Frequency', title = 'Frequency of no. of fives and sixes in 12 throws')

    def binom(k, p):
        val =  stats.binom.pmf(k = k, p = p, n = Ntrials)
        return val

    def poisson(k, mean):
        val =  stats.poisson.pmf(k, mean)
        return val

    def gauss(x, mu, sigma):
        return 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-0.5 * (x-mu) ** 2 / sigma ** 2)


    ## compare to expected distribution. Does it match? YES
    x_vals = np.linspace(0,12,500)
    parameter_guess = np.array([p_naive])
    fit_binom = do_chi2_fit(binom, range[mask], fraction[mask], errors[mask], parameter_guess)
    ax0.plot(range,binom(range,fit_binom.values['p']), '.-', label = r'Binomial fit = $Binom(k,n,p)$')
    d = generate_dictionary(fit_binom, N_non_empty, chi2_suffix = 'Binom')

 
    ## Fit the data and test if alt. hypothesis fit datter better

    fit_poiss = do_chi2_fit(poisson, range[mask], fraction[mask], errors[mask], parameter_guess)
    ax0.plot(range,poisson(range, fit_poiss.values['mean']), '.-', label = r'Poisson fit = $Poiss(k,\lambda)$')
    d0 = generate_dictionary(fit_poiss, N_non_empty, chi2_suffix = 'Poiss')
    d.update(d0)

    parameter_guess = np.array([5.0, 2.0])
    fit_gauss = do_chi2_fit(gauss, range[mask], fraction[mask], errors[mask], parameter_guess)
    ax0.plot(x_vals,gauss(x_vals,*fit_gauss.values[:]), label = 'Gaussian fit = $\mathcal{N}(k,\mu, \sigma^2)$')
    d0 = generate_dictionary(fit_gauss, N_non_empty, chi2_suffix = 'Gauss')
    d.update(d0)


    ## Determine if the dice are consistent with being fair.
    print("From the binomial we, we immidately get the probability of getting a 5 or a 6, namely ", fit_binom.values['p'])
    print("A fair die has p = ", 1/3, " and we to a 1 sample twst to check the consistency")
    t, p = one_sample_test(fit_binom.values['p'], exp_value = 1/3, error_on_mean = fit_binom.errors['p'], small_statistics = False)
    print("test statistics and p-val: ", t, p)

    # Plot figure text    
    text = nice_string_output(d, extra_spacing=0, decimals=3)
    add_text_to_ax(0.65, 0.7, text, ax0, fontsize=13)

    ax0.legend()
    fig0.tight_layout()
    plt.show()



    



    print("sample estiated prop. for at 5 or a 6: ", frequency[1] / frequency.sum(), "\u00B1", \
    np.sqrt(frequency[1] / frequency.sum() * (1 - frequency[1] / frequency.sum()) / frequency.sum()))
    print("theoretical prob. for a fair die: ", 2/36)

def P5_1():
 
    channel = np.loadtxt('data_GammaSpectrum.txt')
    Ndatapoints = len(channel)
    range = (np.min(channel), np.max(channel))
    bins = 1000
    fitting_jump = 10
    bin_width = (range[1] - range[0]) / bins
    bin_jump = int(fitting_jump / bin_width)

    lead_peaks = [242, 295, 352] #(kEV)
    lead_ratios = [7.4, 19.3, 37.6]
    bismuth_peaks = [609, 1120] # (kEV)
    bismith_ratios = [46.1, 15.1]
    lead_peaks_channel = np.array([113, 137.5, 163.9])
    bismuth_peaks_channel = np.array([281, 514], dtype = 'float')


    fig0, ax0 = plt.subplots()

    counts, edges, _ = ax0.hist(channel, bins = bins, range = range, histtype='stepfilled', lw = 2, alpha = .4, label = 'Gamma emission histogram')
    mask = (counts > 0)
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    errors = np.sqrt(counts)
   # ax0.errorbar(bin_centers, counts, errors , fmt = '.', elinewidth=.6, capsize=.6, capthick=.7, markersize = 2)
    ax0.set(xlabel = r'Channel number', ylabel = 'Count', title = 'Gamma ray spectrum of uranium ore')

    lead_bin_peaks = ((lead_peaks_channel-range[0]) / bin_width + 0.5 * bin_width).astype('int')
    bismuth_bin_peaks = ((bismuth_peaks_channel-range[0]) / bin_width + 0.5 * bin_width).astype('int')



    def lin_func(x, a, b):
        return a * x + b

    def gaussian_binned(x, N, mean, std):
        bin_width = (range[1] - range[0]) /  bins
        return  N * bin_width * 1 / (np.sqrt(2 * np.pi) * std) * np.exp(-0.5 * (x-mean) ** 2 / std ** 2)

    def fit_comb(x, a, b, N, mean, std):
        return lin_func(x, a, b) + gaussian_binned(x, N, mean, std)


    ## FIT FOR LEAD
    Npeaks_lead = 3
    param_guess = [np.array([0., 115, 340, lead_peaks_channel[0], 2 ]), np.array([-0.1, 100, 900, lead_peaks_channel[1], 2 ]), \
        np.array([-0.1, 90, 1400, lead_peaks_channel[2], 2 ])]
    pb_fitted_params = np.empty([Npeaks_lead,5])
    pb_fitted_params_error = np.empty([Npeaks_lead,5])
    d = {}
  
    for i, guess in enumerate(param_guess):
        mask1 = ((bin_centers[lead_bin_peaks[i]] - fitting_jump < bin_centers) & (bin_centers[lead_bin_peaks[i]] + fitting_jump > bin_centers))
        Nbins = len(bin_centers[mask1])
    
        fit_pb1 = do_chi2_fit(fit_comb, bin_centers[mask1], counts[mask1], errors[mask1], guess)

        x_vals = np.linspace(lead_peaks_channel[i] - fitting_jump, lead_peaks_channel[i] + fitting_jump, 500)
        ax0.plot(x_vals, fit_comb(x_vals, *fit_pb1.values[:]), label = f'Fit {i} (Pb)', alpha = 0.7)

        Ndof, chi2, p = get_statistics_from_fit(fit_pb1, Nbins)
        d0 = { f"(fit {i}) Ndof": Ndof, f"(fit {i}) Chi2": chi2, f"(fit {i}) Prop": p}
        d.update(d0)
        pb_fitted_params[i] = fit_pb1.values
        pb_fitted_params_error[i] = fit_pb1.errors


    ## FIT FOR BISMUTH
    Npeaks_bismuth = 2
    param_guess = [np.array([0., 20, 500, bismuth_peaks_channel[0], 2 ]), np.array([-0.1, 10, 300, bismuth_peaks_channel[1], 2 ])]
    bi_fitted_params = np.empty([Npeaks_bismuth,5])
    bi_fitted_params_error = np.empty([Npeaks_bismuth,5])
  
    for i, guess in enumerate(param_guess):
        mask1 = ((bin_centers[bismuth_bin_peaks[i]] - fitting_jump < bin_centers) & (bin_centers[bismuth_bin_peaks[i]] + fitting_jump > bin_centers))
        Nbins = len(bin_centers[mask1])
    
        fit_bi = do_chi2_fit(fit_comb, bin_centers[mask1], counts[mask1], errors[mask1], guess)

        x_vals = np.linspace(bismuth_peaks_channel[i] - fitting_jump, bismuth_peaks_channel[i] + fitting_jump, 500)
        ax0.plot(x_vals, fit_comb(x_vals, *fit_bi.values[:]), label = f'Fit {Npeaks_lead + i} (Bi)', alpha = 0.7)

        Ndof, chi2, p = get_statistics_from_fit(fit_bi, Nbins)
        d0 = { f"(fit {Npeaks_lead +i}) Ndof": Ndof, f"(fit {Npeaks_lead +i}) Chi2": chi2, f"(fit {Npeaks_lead +i}) Prop": p}
        d.update(d0)
        bi_fitted_params[i] = fit_bi.values
        bi_fitted_params_error[i] = fit_bi.errors




    energy_rel_dist = (lead_peaks[2] - lead_peaks[1]) / (lead_peaks[1] - lead_peaks[0])
    channel_rel_dist = (pb_fitted_params[2,-2] - pb_fitted_params[1,-2]) / (pb_fitted_params[1,-2] - pb_fitted_params[0,-2])

    ## Write eq above as z = (p3 - p2) / (p2 - p1)
    dzdp3 = lambda p1, p2, p3: 1 / (p2 - p1)
    dzdp2 = lambda p1, p2, p3: -1 / (p2 - p1) - (p3 - p2) / (p2 - p1) ** 2
    dzdp1 = lambda p1, p2, p3:  (p3 - p2) / (p2 - p1) ** 2
 

    channel_rel_dist_error, _, _, _= prop_err_3var(dzdp1, dzdp2, dzdp3, pb_fitted_params[0,-2], pb_fitted_params[1,-2], \
        pb_fitted_params[2,-2], pb_fitted_params_error[0,-2], pb_fitted_params_error[1,-2], pb_fitted_params_error[2,-2])

    print("Energy relative distance: ", energy_rel_dist)
    print("Channel relative distance: ", channel_rel_dist, "\u00B1", channel_rel_dist_error)
   
    print("Perform 1 sample z-test to check if consistent")
    test_statistic, p = one_sample_test(channel_rel_dist, exp_value = energy_rel_dist, error_on_mean = channel_rel_dist_error, small_statistics = False)
    print("test statistic and p-val ", test_statistic, p )
    


    ## Determine the energy scale and whether it is linear --> plot the channel number of the peaks with uncertainties against the known energies

    energies = np.r_['0', np.array(lead_peaks), np.array(bismuth_peaks)]
    channels = np.r_['0', pb_fitted_params[:,-2], bi_fitted_params[:,-2]]
    channels_err = np.r_['0', pb_fitted_params_error[:,-2], bi_fitted_params_error[:,-2]]
    
    fig1, ax1 = plt.subplots()
    print(channels)
    print(channels_err)

    ax1.errorbar(energies, channels, channels_err, fmt = 'k.', elinewidth=1, capsize=1, capthick=1, markersize = 4)
    ax1.set(xlabel = 'Energy (keV)', ylabel = 'Channel number', title = 'Peak energies and channel numbers')

    lin_func = lambda x, a, b: a * (x - 200) + b

    param_guess = np.array([0.5, 100])
    fit_cal = do_chi2_fit(lin_func, energies, channels, channels_err, param_guess)

    ax1.plot(energies, lin_func(energies, *fit_cal.values[:]), label = r'Linear fit = $a(x-200)+b$', alpha = 0.3)
    d3 = generate_dictionary(fit_cal, len(energies))

    text = nice_string_output(d3, extra_spacing=2, decimals=3)
    add_text_to_ax(0.05, 0.9, text, ax1, fontsize=12)

    ax1.legend()
    fig1.tight_layout()

    print("Relationship is clearly linear, albeit the fitted mean errors are so small that P = 0.00. If one makes them thrice as big, P = 0.08")
    print("Probably an indication that some other (unkown) factors are introducing additional uncertainty.")
    print("Energy is now determined as 200 + (channel_number - b) / a")

    ## DO cross check. use calibration to reporduce energy peaks and check that it holds.
    energy_calibration = lambda y: 200 + (y - fit_cal.values['b']) / fit_cal.values['a']

    ## Is the peak width constant or dependent on energy? Do a chi sqaure fit to check
    fig2, ax2 = plt.subplots()

    widths = np.r_['0', pb_fitted_params[:,-1], bi_fitted_params[:,-1]]
    widths_err = np.r_['0', pb_fitted_params_error[:,-1], bi_fitted_params_error[:,-1]]

    const_func = lambda x,a: a
    param_guess = np.array([widths.mean()])
    fit_std = do_chi2_fit(const_func, np.arange(len(widths)), widths, widths_err, param_guess)
   
   
   
    param_guess = np.array([-0.1, widths.mean()])
    lin_func = lambda x, c, b: c * x  + b

    fit_std_lin = do_chi2_fit(lin_func, np.arange(len(widths)), widths, widths_err, param_guess)
    ax2.errorbar(np.arange(len(widths)), widths, widths_err, fmt = 'k.', elinewidth=1, capsize=1, capthick=1, markersize = 4)
    ax2.plot(np.arange(len(widths)), np.ones_like(widths)*fit_std.values['a'], label = r'Const fit = $a$', alpha = 0.3)
    ax2.plot(np.arange(len(widths)),lin_func(np.arange(len(widths)), *fit_std_lin.values[:]), label = r'Lin. fit = $ax + b$', alpha = 0.3)
    d1 = generate_dictionary(fit_std, len(widths), chi2_suffix = 'const.')
    d0 = generate_dictionary(fit_std_lin, len(widths), chi2_suffix = 'lin.')
    d1.update(d0)

    text = nice_string_output(d1, extra_spacing=2, decimals=3)
    add_text_to_ax(0.45, 0.9, text, ax2, fontsize=12)

    ax2.legend()
    fig2.tight_layout()


    #Does the data predict small peak in 700-800 keV range? We'll fit all of them
    unknown_peaks_channel = np.array([342.8, 421, 425.5])
    unknown_bin_peaks = ((unknown_peaks_channel-range[0]) / bin_width + 0.5 * bin_width).astype('int')

    


     ## FIT FOR LAST 3 PEAKS
    Npeaks_unknown = 3
    param_guess = [np.array([0., 20, 100, unknown_peaks_channel[0], .8 ]), np.array([-0.1, 20, 100, unknown_peaks_channel[1], .8 ]), \
        np.array([-0.1, 20, 100, unknown_peaks_channel[2], .8 ])]
    unk_fitted_params = np.empty([Npeaks_unknown,5])
    unk_fitted_params_error = np.empty([Npeaks_unknown,5])
    d2 = {}
  
    for i, guess in enumerate(param_guess):
        if i>=1:
            fitting_jump = 3
        mask1 = ((bin_centers[unknown_bin_peaks[i]] - fitting_jump < bin_centers) & (bin_centers[unknown_bin_peaks[i]] + fitting_jump > bin_centers))
        Nbins = len(bin_centers[mask1])
    
        fit_unk = do_chi2_fit(fit_comb, bin_centers[mask1], counts[mask1], errors[mask1], guess)

        x_vals = np.linspace(unknown_peaks_channel[i] - fitting_jump, unknown_peaks_channel[i] + fitting_jump, 500)
        ax0.plot(x_vals, fit_comb(x_vals, *fit_unk.values[:]), label = f'Fit {i + Npeaks_bismuth + Npeaks_lead}', alpha = 0.7)

        Ndof, chi2, p = get_statistics_from_fit(fit_unk, Nbins)
        d0 = { f"(fit {i+ Npeaks_bismuth + Npeaks_lead}) Ndof": Ndof, f"(fit {i+ Npeaks_bismuth + Npeaks_lead}) Chi2": chi2, f"(fit {i+ Npeaks_bismuth + Npeaks_lead}) Prop": p}
        d2.update(d0)
        unk_fitted_params[i] = fit_unk.values
        unk_fitted_params_error[i] = fit_unk.errors

    

    text = nice_string_output(d, extra_spacing=2, decimals=3)
    text2 = nice_string_output(d2, extra_spacing=2, decimals=3)
    add_text_to_ax(0.4, 0.9, text, ax0, fontsize=12)
    add_text_to_ax(0.05, 0.9, text2, ax0, fontsize=12)

    ax0.legend()
    fig0.tight_layout()

    print("Energies of fits: ", energy_calibration(unk_fitted_params[:,-2]))
    print("here, you should calculate uncertainties as well.")

    print("let us finally consider distance of normalization constants from 0 for each peaks")
    norm_peaks = np.r_['0', pb_fitted_params[:,-3], bi_fitted_params[:,-3], unk_fitted_params[:,-3]]
    norm_peaks_err = np.r_['0', pb_fitted_params_error[:,-3], bi_fitted_params_error[:,-3], unk_fitted_params_error[:,-3]]

    norm_ratios = norm_peaks / norm_peaks_err
    gauss_prop = stats.norm.sf(norm_ratios)
    peak_no = np.arange(len(norm_ratios))

    arr = np.block([[peak_no],[norm_ratios],[gauss_prop]]).T

    print("Peak no.   N/dN       P(N = 0)")
    print(arr)


    plt.show()

def P5_2():
    pass
  


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
rcParams['axes.prop_cycle'] = cycler(color = ['teal', 'navy', 'coral', 'plum', 'purple', 'olivedrab', 'black', 'red', 'cyan', 'yellow', 'khaki','lightblue'])
np.set_printoptions(precision = 5, suppress=1e-10)

## Set parameters and which problems to run
p1_1, p1_2, p1_3  = False, False, False
p2_1, p2_2, p3_1, p3_2 = False, False, True, False
p4_1, p4_2, p5_1, p5_2 =  False, False, False, False


def main():
    
    problem_numbers = [p1_1, p1_2, p1_3, p2_1, p2_2, p3_1, p3_2, p4_1, p4_2, p5_1, p5_2]
    f_list = [P1_1, P1_2, P1_3, P2_1, P2_2, P3_1, P3_2, P4_1, P4_2, P5_1, P5_2]
    names = ['p1_1', 'p1_2', 'p1_3', 'p2_1', 'p2_2', 'p3_1', 'p3_2', 'p4_1', 'p4_2', 'p5_1', 'p5_2']

    for i, f in enumerate(f_list):
        if problem_numbers[i]:
            print(f'\nPROBLEM {names[i][1:]}:')
            f()
   

if __name__ == '__main__':
    main()


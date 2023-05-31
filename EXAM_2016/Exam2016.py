# Author: Simon Guldager Andersen
# Date (latest update): 

### SETUP -----------------------------------------------------------------------------------------------------------------------------------

## Imports:
import os, sys
import numpy as np
import iminuit
import seaborn as sns
import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib import rcParams
from scipy import stats, integrate, optimize

sys.path.append('Appstat2022\\External_Functions')
from ExternalFunctions import Chi2Regression, BinnedLH, UnbinnedLH
from ExternalFunctions import nice_string_output, add_text_to_ax    # Useful functions to print fit results on figure


## Change directory to current one
os.chdir('AppStat2022\\EXAM_2016')


### FUNCTIONS -------------------------------------------------------------------------------------------------------------------------------

def generate_dictionary(fitting_object, Ndatapoints, chi2_fit = True, chi2_suffix = None):

    Nparameters = len(fitting_object.values[:])
    dictionary = {'Entries': Ndatapoints}

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

def one_sample_test(sample_array, exp_value, one_sided = False, small_statistics = False):
    """ Assuming that the errors to be used are the standard error on the mean as calculated by the sample std 
    Returns test-statistic, p_val
    """
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

def calc_ROC(hist1, hist2, input_is_hist = True, bins = None, range = None) :
    """
    Calculate ROC curve from two histograms (hist1 is signal, hist2 is background):
    if input_is_hist = False, the input entries are assume to be arrays
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
         
        return FPR, TPR
    
    else:
        AssertionError("Signal and Background histograms have different bins and/or ranges")

def calc_sample_purity(hist1, hist2) :

    # First we extract the entries (y values) and the edges of the histograms:
    # Note how the "_" is simply used for the rest of what e.g. "hist1" returns (not really of our interest)
    y_sig, x_sig_edges, _ = hist1 
    y_bkg, x_bkg_edges, _ = hist2
    
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
                SP[i] = sig_area / (sig_area + bkg_area)                     # False positive rate            
        return SP
    else:
        AssertionError("Signal and Background histograms have different bins and/or ranges")


### TEMPLATES ------------------------------------------------------------------------------------------------------------------------------------

## FITTING FUNCTIONS
def gaussian_binned(x, N, mean, std):
    bin_width = (range[1] - range[0]) /  bins
    return N * bin_width * 1 / (np.sqrt(2 * np.pi) * std) * np.exp(-0.5 * (x-mean) ** 2 / std ** 2)

def double_gaussian_binned(x, N1, N2, mean1, mean2, std1, std2):
    bin_width = (range[1] - range[0]) /  bins
    val1 = N1 * bin_width * 1 / (np.sqrt(2 * np.pi) * std1) * np.exp(-0.5 * (x-mean1) ** 2 / std1 ** 2)
    val2 = N2 * bin_width * 1 / (np.sqrt(2 * np.pi) * std2) * np.exp(-0.5 * (x-mean2) ** 2 / std2 ** 2)
    return val1 + val2
        
def gaussian_LH(x, N, mean, std):
    return  N * 1 / (np.sqrt(2 * np.pi) * std) * np.exp(-0.5 * (x-mean) ** 2 / std ** 2)

def double_gaussian_LH(x, N1, N2, mean1, mean2, std1, std2):

    val1 = N1 * 1 / (np.sqrt(2 * np.pi) * std1) * np.exp(-0.5 * (x-mean1) ** 2 / std1 ** 2)
    val2 = N2 * 1 / (np.sqrt(2 * np.pi) * std2) * np.exp(-0.5 * (x-mean2) ** 2 / std2 ** 2)
    return val1 + val2

    
## Calc. mean, std, SEM and compare with smirnov and t-test
def template_means_and_test():

      # Initialize data
    drug_group =  np.array([3.7, -1.2, -0.2, 0.7, 0.8])
    control_group = np.array([1.5, -1.0, -0.7, 0.5, 0.1])


    ## Estimate mean, (unbiased) std and error on the mean for each group
    groups = [drug_group, control_group]
    names = ['drug group', 'control group']

    for i, group in enumerate(groups):
        mean, std = group.mean(), group.std(ddof = 1)
        err_mean = std / np.sqrt(len(group))

        print(f'\nFor {names[i]}:')
        print(f'Mean, Std, Error on mean: ', mean, " ", std, " ", err_mean)

    
    ## Examine combatibility with Kolmogrov test under the null hypothesis that the cumulative distribution of the drug group if shifted
    # towards larger values (so that F_drug(x) <= F_control(x))
    _, p_val = stats.ks_2samp(drug_group, control_group, alternative = 'less')

    print('\np-value of the hypothesis that the cumulative sleep distribution of the drug group is shifted towards larger values: ', p_val)

    # Use t-test to calculate p_value under the null hypothesis that the 2 means are equal, letting the alternative hypothesis be that the mean of the drug group
    # is greater than that of the control group
    t_val, p_val = stats.ttest_ind(drug_group, control_group, equal_var = True, alternative = 'greater')

    print("\n t_val and Probability that the drug group and control group means are equal (using a one-tailed 2 sample t-test)", t_val, p_val)

## Calc error prop of 2 variables
def template_error_prop_2_var():
       ## Initialize variables
    x, dx = 1.92, 0.39
    y, dy = 3.1, 1.3

    # z1 = y/x
    dz1dx = lambda x, y: - y / x ** 2
    dz1dy = lambda x, y: 1/x

    # z2 = cos(x) * x/y
    dz2dx = lambda x,y: np.cos(x) / y - np.sin(x) * x / y
    dz2dy = lambda x, y: - np.cos(x) * x / y ** 2
 
    ## Find uncertainties of z1 and z2 if uncorrelated
    err_z1, err_z1_from_x, err_z1_from_y = prop_err(dz1dx, dz1dy, x, y, dx, dy)
    err_z2, err_z2_from_x, err_z2_from_y = prop_err(dz2dx, dz2dy, x, y, dx, dy)

    print("\nFor z1: ")
    print(f'Total uncertainty: {err_z1},  error propagated from x: {err_z1_from_x},  error propagated from y: {err_z1_from_y}')
    print("\nFor z2: ")
    print(f'Total uncertainty: {err_z2},  error propagated from x: {err_z2_from_x},  error propagated from y: {err_z2_from_y}')

    ## Find uncertainties if correlation coefficient = 0.95
    correlation = 0.95
    
    err_z1 = prop_err(dz1dx, dz1dy, x, y, dx, dy, correlation = correlation)
    err_z2 = prop_err(dz2dx, dz2dy, x, y, dx, dy, correlation = correlation)

    print(f'Error on z1 if correlation = 0.95:  {err_z1}')
    print(f'Error on z2 if correlation = 0.95:  {err_z2}') 

## Transformation + Rejection sampling, fitting and MC plotting
def template_MC_plot_and_fit():
        ## Consider distribution f = C * x ** (-0.9) from x in [0.005,1]

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
    Npoints = 10_000
    uniform_vals = np.random.rand(10_000)

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
    plt.show()


def template_plot_fit_histogram():
     ### Fit distribution with Gaussian

    fig, ax = plt.subplots()
    range = (np.min(residuals), np.max(residuals))
    bins = 80

    def gaussian_binned(x, N, mean, std):
        bin_width = (range[1] - range[0]) /  bins
        return N * bin_width * 1 / (np.sqrt(2 * np.pi) * std) * np.exp(-0.5 * (x-mean) ** 2 / std ** 2)

    def gaussian_LH(x, N, mean, std):
        return  N * 1 / (np.sqrt(2 * np.pi) * std) * np.exp(-0.5 * (x-mean) ** 2 / std ** 2)


    counts, edges, _ = ax.hist(residuals, bins = bins, range = range, histtype = 'stepfilled', lw = 2, alpha = 0.5, label = 'Histogram of time residuals')
    
    x_vals = 0.5 * (edges[:-1] + edges[1:])
    errors = np.sqrt(counts)
    mask = (counts > 0)
    N_non_empty = len(mask)

    ax.set(xlabel = 'Time residual (s)', ylabel = 'Count')
    ax.errorbar(x_vals, counts, errors, fmt = 'k.', elinewidth=1, capsize=1, capthick=1)

    parameter_guesses = np.array([1500, 0, 0.2])
    fit_chi2 = do_chi2_fit(gaussian_binned, x_vals[mask], counts[mask], errors[mask], parameter_guesses)
    fit_LH = do_LH_fit(gaussian_LH, residuals , fit_chi2.values, bound = range, unbinned = True, extended = True)

    if 1:
        fmax = gaussian_LH(fit_LH.values['mean'], *fit_LH.values)
        Nsimulations = 100
        LL_values, p_val = evaluate_likelihood_fit(gaussian_LH, fmax = fmax, parameter_val_arr = fit_LH.values, \
            log_likelihood_val = fit_LH.fval, bounds = range, Ndatapoints = len(residuals), Nsimulations = Nsimulations)

        plot_likelihood_fits(LL_values, p_val, log_likelihood_val = fit_LH.fval, Nsimulations = Nsimulations)
  
    x_fit = np.linspace(range[0], range[1], 1000)
    fit_vals_chi2 = gaussian_binned(x_fit, *fit_chi2.values[:])
    fit_vals_LH = gaussian_binned(x_fit, *fit_LH.values[:])
    ax.plot(x_fit, fit_vals_LH, label = 'LH Gaussian fit')
    ax.plot(x_fit, fit_vals_chi2, label = r'Chi2 Gaussian fit')
    
    Ndof, chi2, p = get_statistics_from_fit(fit_chi2, N_non_empty)
    d = {"Gaussian (chi2) fit:": " ","Ndof": Ndof, "chi2": chi2, 'Prop': p}
    #d = generate_dictionary(fit_chi2, Ndatapoints = N_non_empty, chi2_suffix = 'chi2')

    ### Fit with other PDFs. Compare to Gaussian fit.
    text = nice_string_output(d, extra_spacing=0, decimals=2)
    add_text_to_ax(0.05, 0.9, text, ax, fontsize=14)

    ax.legend()
    fig.tight_layout()
    plt.show()

def template_one_plots_fit_points():
        ## Load data x is time in month, y is monthly income in M$, dy is error on income.
    x, y, dy = np.loadtxt('data_LukeLightningLights.txt', usecols = (0,2,3), unpack = True)
    assert (x.shape == y.shape == dy.shape)

    ### STEP 1: Plot the thing
    fig0, ax0 = plt.subplots()
    ax0.errorbar(x, y, dy, fmt = 'k.', elinewidth=1.5, capsize=1.5, capthick=1)
    ax0.set(xlabel = 'Time (months)', ylabel = 'Monthly income (M$)', title = "LLL's monthly income over time")
    fig0.tight_layout()
   

    ### Was monthly income constant over the first 12 months?
    index_year = 12
    print("Using significance level of 0.01")
    def fit_func(x,a):
        return a

    def fit_func_lin(x, c, d):
        return c * x + d


    fit = do_chi2_fit(fit_func, x[:index_year], y[:index_year], dy[:index_year], parameter_guesses= np.array([-0.35]), verbose = True)
    fit_lin = do_chi2_fit(fit_func_lin, x[:index_year], y[:index_year], dy[:index_year], parameter_guesses= np.array([0.2, -0.4]))

    x_constant = np.linspace(0,12,10)
    ax0.plot(x_constant, fit_func(x_constant, fit.values['a'] * np.ones_like(x_constant)), label = r"Const. fit = a")
    ax0.plot(x_constant, fit_func_lin(x_constant, *fit_lin.values[:]), label = r"Lin. fit = cx + d")

    d0 = generate_dictionary(fit, Ndatapoints = len(y[:index_year]), chi2_suffix = 'const')
    d = generate_dictionary(fit_lin, Ndatapoints = len(y[:index_year]), chi2_suffix = 'lin')
    d.update(d0)

    # Plot figure text
    text = nice_string_output(d, extra_spacing=0, decimals=2)
    add_text_to_ax(0.05, 0.7, text, ax0, fontsize=13)


### TO BE IMPROVED UPON



### MAIN ------------------------------------------------------------------------------------------------------------------------------------

def P1_1():
    ## FIND p for getting at least one 6 in 4 throws
    p1 = stats.binom.sf(k = 0, n = 4, p = 1 / 6)
    ## FIND p for getting (6,6) at least one time in 24 throws
    p2 = stats.binom.sf(k = 0, n = 24, p = 1/36)

    print("P(k >= 1, n = 4, p = 1/6) = ", p1)
    print("P(k >= 1, n = 24, p = 1/36) = ", p2)
    

def P1_2():

    Ndays = 1730
    mean_rate = 18.9

    ## what distribution should the daily number of background events follow? == Poisson

    ## If the experiment in a day saw 42 events, would that signify a significant excess?
    significance_level = 0.01
    print("We shall use a significance level of ", significance_level)

    
    p_local = stats.poisson.sf(k = 41, mu = mean_rate)
    print("The probability that any one day has 42 events or more is ", p_local)

    print("I we are not just looking at one day's data, but are considering whether this event ocurred for any of the ", Ndays, " days, the global prob. is")
    p_global = 1 - (1 - p_local) ** Ndays
    print(p_global)
    

def P1_3():
    ## Assume heigh of Danish women follow a Gaussian distribution with 
    mean, std = 1.68, 0.06  # (m)

    ## What fraction of women are taller than 1.85 m ?
    x_cutoff = 1.85
    fraction = stats.norm.sf(x_cutoff, loc = mean, scale = std)
    print("What fraction of women are taller than 1.85 m ? ", fraction)

    ## Find the average height of the 20% tallest women
    p_cutoff = 0.200000000
    x_cutoff = stats.norm.isf(p_cutoff, loc = mean, scale = std)
    print("80'th height percentile: ", x_cutoff)

    ## To find av. height, gen. 20_000_000 gaussian values from and take the average of the 20% greatest. Do loop to extract uncertainty
    Nsamples = 20_000_000
    Nloops = 15
    av_list = np.empty(Nloops)
    for i in range(Nloops):
        vals = stats.norm.rvs(loc = mean, scale = std, size = Nsamples)
        mask = (vals > x_cutoff)
        av_list[i] = vals[mask].mean()


    print("av height of 20% tallest: ", av_list.mean(), "\u00B1", av_list.std(ddof = 1) / np.sqrt(Nloops) )


def P2_1():
    ## Initialize variables
    x, dx = 1.92, 0.39
    y, dy = 3.1, 1.3

    # z1 = y/x
    dz1dx = lambda x, y: - y / x ** 2
    dz1dy = lambda x, y: 1/x

    # z2 = cos(x) * x/y
    dz2dx = lambda x,y: np.cos(x) / y - np.sin(x) * x / y
    dz2dy = lambda x, y: - np.cos(x) * x / y ** 2
 
    ## Find uncertainties of z1 and z2 if uncorrelated
    err_z1, err_z1_from_x, err_z1_from_y = prop_err(dz1dx, dz1dy, x, y, dx, dy)
    err_z2, err_z2_from_x, err_z2_from_y = prop_err(dz2dx, dz2dy, x, y, dx, dy)

    print("\nFor z1: ")
    print(f'Total uncertainty: {err_z1},  error propagated from x: {err_z1_from_x},  error propagated from y: {err_z1_from_y}')
    print("\nFor z2: ")
    print(f'Total uncertainty: {err_z2},  error propagated from x: {err_z2_from_x},  error propagated from y: {err_z2_from_y}')

    ## Find uncertainties if correlation coefficient = 0.95
    correlation = 0.95
    
    err_z1 = prop_err(dz1dx, dz1dy, x, y, dx, dy, correlation = correlation)
    err_z2 = prop_err(dz2dx, dz2dy, x, y, dx, dy, correlation = correlation)

    print(f'Error on z1 if correlation = 0.95:  {err_z1}')
    print(f'Error on z2 if correlation = 0.95:  {err_z2}') 


def P2_2():
    # Initialize data

    speeds = 1e2 * np.array([3.61, 2.00, 3.9, 2.23, 2.32, 2.48, 2.43, 3.86, 4.43, 3.78]) # m/s

    mean, std, ems = calc_mean_std_sem(speeds)

    print("mean, std, ems: ", mean, std, ems)

    ## Initialize variables

    #MASS
    x, dx = 8.4 * 1e-3, 0.5 * 1e-3 # kg
    #SPEED
    v, dv = mean, ems

    # z = 0.5 m v^2
    dzdv = lambda x, v: x * v 
    dzdx = lambda x, v: 0.5 * v ** 2


    ## Find uncertainties of z1 and z2 if uncorrelated
    err_z1, err_z1_from_x, err_z1_from_y = prop_err(dzdx, dzdv, x, v, dx, dv)

    err_v = np.sqrt(dzdv(x,v) ** 2 * dv ** 2)
    err_m = np.sqrt(dzdx(x,v) ** 2 * dx ** 2)
    print(err_m, err_v)

    print("\nFor E: ")
    print(f'Total uncertainty: {err_z1:.6e},  error propagated from mass: {err_z1_from_x:.6e},  error propagated from speed: {err_z1_from_y}')
 
    N_equal_unc = dzdv(x,v) ** 2 * std ** 2 / err_z1_from_x ** 2
    print("needed v-measurements for equal uncertainty: ", N_equal_unc)


def P3_1():
     ## Consider distribution f = C * x ** (-0.9) from x in [0.005,1]

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
   pass

def P4_1():
    ## Load data, 1st column records health label (0 = healthy, 1 = ill), last 3 columns report concentrations of substances A,B,C in blood
    labels, dataA, dataB, dataC = np.loadtxt('data_FisherSyndrome.txt', usecols = (0,2,3,4), unpack = True)
    assert (labels.shape == dataA.shape == dataB.shape == dataC.shape)

    indices_ill = np.argwhere(labels == 1).flatten()
    indices_healthy = np.argwhere(labels == 0).flatten()

    ### What distributions does A seem to follows for ill people?
    dataA_ill = dataA[indices_ill]
    dataB_ill = dataB[indices_ill]
    dataC_ill = dataC[indices_ill]

    dataA_healthy = dataA[indices_healthy]
    dataB_healthy = dataB[indices_healthy]
    dataC_healthy = dataC[indices_healthy]

    data_ill = np.block([[dataA_ill],[dataB_ill],[dataC_ill]]).T
    data_healthy = np.block([[dataA_healthy],[dataB_healthy],[dataC_healthy]]).T

    # TRY TO PLOT IT
    fig1, ax1 = plt.subplots()
    bins = 100

    counts, edges, _ = ax1.hist(dataA_ill, bins = bins, histtype = 'step', lw = 2, label = 'Histogram of substance A values for ill people')
    x_vals = 0.5 * (edges[:-1] + edges[1:])
    count_err = np.sqrt(counts)
    ax1.errorbar(x_vals, counts, count_err, fmt = 'k.', elinewidth=1.5, capsize=1.5, capthick=1)
    ax1.legend()
    plt.show(block = False)

    ## LOOKS like a normal distribution. Let's test it with Anderson's against a normal distribution with unspecified mean and variance
    res_anderson = stats.anderson(dataA_ill, dist = 'norm')

    print("Applying the Anderson test to see if the distribution of A for ill people follows a normal distribution results in:")
    print("test statistic:", res_anderson.statistic)
    print("critical values of test statistics:", res_anderson.critical_values)
    print("corresponding significance levels: ", res_anderson.significance_level)
    print("since the test statistics is less than the critical value at significant 15 %, even at 15% the null hypothesis \
        that the distribution follows a normal distribution cannot be rejected.")

    print("Testing the same thing with 2-sided Smirnov (with mean and std equal to distribution):")
    
    res_smirnov = stats.kstest(dataA_ill, lambda x: stats.norm.cdf(x, loc = dataA_ill.mean(), scale = dataA_ill.std(ddof = 1)))
    print("Smirnov test statistic and p value: ", res_smirnov[0], res_smirnov[1])
    print(res_smirnov)

    ## What is the linear correlation between B and C for ill people?
    BC_matrix = np.r_['0,2', dataB_ill, dataC_ill].T
   
    corr_BC = calc_corr_matrix(BC_matrix)
    print("BC corr. matrix (ddof = 1) \n", corr_BC)
    print("NP Corr", np.corrcoef(dataB_ill, dataC_ill))

    ## Find the best separation between healthy and ill people using any variable of combination
    fig2,ax2 = plt.subplots()
    fig3, ax3 = plt.subplots()
    fig4, ax4 = plt.subplots()
    bins = 50

    list_healthy = [dataA_healthy, dataB_healthy, dataC_healthy]
    list_ill = [dataA_ill, dataB_ill, dataC_ill]
    list_names = ['A', 'B', 'C']

    list_diff_healthy = [dataA_healthy - dataB_healthy, dataA_healthy - dataC_healthy, dataB_healthy - dataC_healthy]
    list_diff_ill = [dataA_ill - dataB_ill, dataA_ill - dataC_ill, dataB_ill - dataC_ill]
    list_diff_names = ['A-B', 'A-C', 'B-C']

    list_sum_healthy = [dataA_healthy + dataB_healthy, dataA_healthy + dataC_healthy, dataB_healthy + dataC_healthy]
    list_sum_ill = [dataA_ill + dataB_ill, dataA_ill + dataC_ill, dataB_ill + dataC_ill]
    list_sum_names = ['A+B', 'A+C', 'B+C']

    for i, (healthy, ill) in enumerate(zip(list_healthy, list_ill)):
        hist_healthy = ax2.hist(healthy, bins = bins, histtype = 'step', lw = 2, label = f'Healthy {list_names[i]}')
        hist_ill = ax2.hist(ill, bins = bins, histtype = 'step', lw = 2, label = f'Ill {list_names[i]}')
     

        hist_diff_healthy = ax3.hist(list_diff_healthy[i], bins = bins, histtype = 'step', lw = 2, label = f' Healthy {list_diff_names[i]}')
        hist_diff_ill = ax3.hist(list_diff_ill[i], bins = bins, histtype = 'step', lw = 2, label = f' Ill {list_diff_names[i]}')

        hist_sum_healthy = ax3.hist(list_sum_healthy[i], bins = bins, histtype = 'step', lw = 2, label = f' Healthy {list_sum_names[i]}')
        hist_sum_ill = ax3.hist(list_sum_ill[i], bins = bins, histtype = 'step', lw = 2, label = f' Ill {list_sum_names[i]}')

        range = (min(np.min(healthy), np.min(ill)), max(np.max(healthy), np.max(ill)))
        range_diff = (min(np.min(list_diff_healthy[i]), np.min(list_diff_ill[i])), max(np.max(list_diff_healthy[i]), np.max(list_diff_ill[i])))
        range_sum = (min(np.min(list_sum_healthy[i]), np.min(list_sum_ill[i])), max(np.max(list_sum_healthy[i]), np.max(list_sum_ill[i])))
       
        if healthy.mean() > ill.mean():
            signal = 'Healthy'
            FPR, TPR = calc_ROC(healthy, ill , input_is_hist = False, bins = 2* bins, range = range)
        else:
            signal = 'Ill'
            FPR, TPR = calc_ROC(ill, healthy , input_is_hist = False, bins = 2*bins, range = range)
        if list_diff_healthy[i].mean() > list_diff_ill[i].mean():
            signal_diff = 'Healthy'
            FPR_diff, TPR_diff = calc_ROC(list_diff_healthy[i], list_diff_ill[i], input_is_hist = False, bins = 2*bins, range = range_diff)
        else:
            signal_diff = 'Ill'
            FPR_diff, TPR_diff = calc_ROC(list_diff_ill[i], list_diff_healthy[i], input_is_hist = False, bins = 2*bins, range = range_diff)
        if list_sum_healthy[i].mean() > list_sum_ill[i].mean():
            signal_diff = 'Healthy'
            FPR_sum, TPR_sum = calc_ROC(list_sum_healthy[i], list_sum_ill[i], input_is_hist = False, bins = 2*bins, range = range_sum)
        else:
            signal_diff = 'Ill'
            FPR_sum, TPR_sum = calc_ROC(list_sum_ill[i], list_sum_healthy[i], input_is_hist = False, bins = 2*bins, range = range_sum)

        ax4.plot(FPR, TPR, label = f'{list_names[i]}: Signal = {signal}')
        ax4.plot(FPR_diff, TPR_diff, label = f'{list_diff_names[i]}: Signal = {signal_diff}')
        ax4.plot(FPR_sum, TPR_sum, label = f'{list_sum_names[i]}: Signal = {signal}')
      

    hist_diff_all_healthy = ax3.hist(dataA_healthy - dataB_healthy - dataC_healthy, range = (-64,20), bins = 2*bins, histtype = 'step', lw = 2, label = 'Healthy A-B-C')
    hist_diff_all_ill = ax3.hist(dataA_ill - dataB_ill - dataC_ill, bins = 2*bins, range = (-64,20), histtype = 'step', lw = 2,  label = 'Ill A-B-C')
    FPR, TPR = calc_ROC(hist_diff_all_ill, hist_diff_all_healthy)
    ax4.plot(FPR, TPR, label = f'A-B-C: Signal = Ill')
    ax4.set(xlabel = r'FPR = $\beta$', ylabel = r'TPR = 1 - $\alpha$', title = 'ROC curves for various combinations of A, B, C', xlim = (-0.1,1))

    
    print("Combining all variables in a fisher discrimminant yields")
    fisher_ill, fisher_healthy = calc_fisher_discrimminant(data_ill, data_healthy, weight_normalization =  5000)
    fig5, ax5 = plt.subplots()
    range_fisher = (min(np.min(fisher_healthy), np.min(fisher_ill)), max(np.max(fisher_healthy), np.max(fisher_ill)))
    hist_fisher_ill = ax5.hist(fisher_ill, bins = 2*bins, range = range_fisher, histtype = 'step', lw = 2, label = 'Fisher ill')
    hist_fisher_healthy = ax5.hist(fisher_healthy, bins = 2*bins, range = range_fisher, histtype = 'step', lw = 2, label = 'Fisher healthy')

    FPR_fisher, TPR_fisher = calc_ROC(hist_fisher_ill, hist_fisher_healthy)
    ax4.plot(FPR_fisher, TPR_fisher, label = f'Fisher: Signal = Ill')
    alpha = 0.01
    critical_indices = np.argwhere(np.abs(TPR_fisher - (1-alpha)) < 0.005).flatten()
    critical_index = np.max(critical_indices)
    _, edges, _ = hist_fisher_ill
    x_vals = 0.5 * (edges[:-1] + edges[1:])
    print("using alpha = ", 1- TPR_fisher[critical_index], "implies beta = ", FPR_fisher[critical_index], " and a concentration cutoff of ", x_vals[critical_index])
  
    ax4.plot([-0.1,FPR_fisher[critical_index]], [TPR_fisher[critical_index], TPR_fisher[critical_index]], 'k--', label = 'alpha = 0.01')
    ax5.legend()
    ax4.legend()
    ax3.legend()
    ax2.legend()
    plt.show()

def P4_2():
    pass


def P5_1():
    ## Load data x is time in month, y is monthly income in M$, dy is error on income.
    x, y, dy = np.loadtxt('data_LukeLightningLights.txt', usecols = (0,2,3), unpack = True)
    assert (x.shape == y.shape == dy.shape)

    ### STEP 1: Plot the thing
    fig0, ax0 = plt.subplots()
    ax0.errorbar(x, y, dy, fmt = 'k.', elinewidth=1.5, capsize=1.5, capthick=1)
    ax0.set(xlabel = 'Time (months)', ylabel = 'Monthly income (M$)', title = "LLL's monthly income over time")
    fig0.tight_layout()
   

    ### Was monthly income constant over the first 12 months?
    index_year = 12
    print("Using significance level of 0.01")
    def fit_func(x,a):
        return a

    def fit_func_lin(x, c, d):
        return c * x + d


    fit = do_chi2_fit(fit_func, x[:index_year], y[:index_year], dy[:index_year], parameter_guesses= np.array([-0.35]), verbose = True)
    fit_lin = do_chi2_fit(fit_func_lin, x[:index_year], y[:index_year], dy[:index_year], parameter_guesses= np.array([0.2, -0.4]))

    x_constant = np.linspace(0,12,10)
    ax0.plot(x_constant, fit_func(x_constant, fit.values['a'] * np.ones_like(x_constant)), label = r"Const. fit = a")
    ax0.plot(x_constant, fit_func_lin(x_constant, *fit_lin.values[:]), label = r"Lin. fit = cx + d")

    d0 = generate_dictionary(fit, Ndatapoints = len(y[:index_year]), chi2_suffix = 'const')
    d = generate_dictionary(fit_lin, Ndatapoints = len(y[:index_year]), chi2_suffix = 'lin')
    d.update(d0)

    # Plot figure text
    text = nice_string_output(d, extra_spacing=0, decimals=2)
    add_text_to_ax(0.05, 0.7, text, ax0, fontsize=13)


    ### Do a chi**2 fit on lin. relation between x and y for first 12 months. How much can you extend it?
    # we are going to do a linear fit while extending the range for up to 20 months
    cutoff = np.arange(12,20)
    p_vals = np.empty_like(cutoff)

    for i, last_month in enumerate(cutoff):
        chi2_object_lin = Chi2Regression(fit_func_lin, x[:last_month], y[:last_month], dy[:last_month])
        fit_lin = iminuit.Minuit(chi2_object_lin, c = 0.2, d = - 0.4)
        fit_lin.errordef = iminuit.Minuit.LEAST_SQUARES
        fit_lin.migrad()
        print(f'last month {last_month}, valid fit ', fit_lin.fmin.is_valid)

        Ndof_lin = len(y[:last_month]) - len(fit_lin.values[:])
        chi2_lin = fit_lin.fval
        prop_lin = stats.chi2.sf(chi2_lin, Ndof_lin)
        p_vals[i] = prop_lin
        print(prop_lin)

    ### The income fell after the 31st month. Estimate how much and uncertainty
    ## NEW FIGURE RINSE AND REPEAT
    cutoff1, cutoff2 = 25, 37
    mask = np.arange(cutoff1, cutoff2)


    def fit_func_lin2(x, a1, b1, dely, a2):
        if x < cutoff1 or x > cutoff2:
            return 0
        elif cutoff1 <= x <= 31:
            return a1 * (x-31) + b1
        elif 31 < x <= cutoff2:
            return a2 * (x-32) + (b1 - dely)

  
    fit_lin2 = do_chi2_fit(fit_func_lin2, x[mask], y[mask], dy[mask], np.array([0.2, 3.9, 0.8, 0.2]))

    lin2_vec = np.vectorize(lambda x: fit_func_lin2(x, *fit_lin2.values[:]))
    vals1 = np.arange(cutoff1, 32)
    vals2 = np.arange(32,cutoff2)
    ax0.plot(vals1, lin2_vec(vals1), label = r"Lin. fit 2 ($x \leq 31$): $a_1(x-31)+b_1$")
    ax0.plot(vals2, lin2_vec(vals2), label = r"Lin. fit 2 ($x \geq 32$): $a_2(x-32)+b_1-\Delta y$")

    Ndof = len(mask) - len(fit_lin2.values[:])
    chi2 = fit_lin2.fval
    p_val = stats.chi2.sf(chi2, Ndof)


   
    d2 = generate_dictionary(fit_lin2, Ndatapoints = len(mask), chi2_suffix = 'lin2')
    # Plot figure text
    text = nice_string_output(d2, extra_spacing=2, decimals=2)
    add_text_to_ax(0.65, 0.5, text, ax0, fontsize=13)
    ax0.legend()


    ### Try to fit entire range with 1 or more hypotheses
    fig1, ax1 = plt.subplots()
    ax1.errorbar(x, y, dy, fmt = 'k.', elinewidth=1.5, capsize=1.5, capthick=1)
    ax1.set(xlabel = 'Time (months)', ylabel = 'Monthly income (M$)', title = "LLL's monthly income over time")

    cutoff1 = 4
    cutoff2 = 32
    
    def fit_func_full(x, y0, A, M, d3, a3, b3):
        
        c3 = (M) / (y0 - A) - 1
        if x < cutoff2:   
            return A + M / (1 + c3 * np.exp(-d3 * (x-cutoff1)))
        elif x >= cutoff2:
            return A + M / (1 + c3 * np.exp(-d3 * cutoff2)) - fit_lin2.values['dely'] + a3 * (x-cutoff2) ** 2 + b3 * (x-cutoff2)


    chi2_object_full = Chi2Regression(fit_func_full, x, y, dy)
    chi2_object_full.errordef = iminuit.Minuit.LEAST_SQUARES
    #fit_full = iminuit.Minuit(chi2_object_full, k = -0.36, h = 0.02, A = -0.1, M = 3.9, d3 = 0.05, a3 = -0.011, b3 = 0.135)
    fit_full = iminuit.Minuit(chi2_object_full, y0 = -0.3, A = -0.1, M = 3.9, d3 = 0.05, a3 = -0.011, b3 = 0.135)
    print(fit_full.migrad())

    Ndof = len(x) - len(fit_full.values[:])
    chi2 = fit_full.fval
    prop = stats.chi2.sf(chi2, Ndof)


    fit_full_vec = np.vectorize(lambda x: fit_func_full(x, *fit_full.values[:]))

    x_vals1 = np.linspace(x[0], cutoff2, 500)
    x_vals2 = np.linspace(cutoff2,x[-1], 100)

    ax1.plot(x_vals1, fit_full_vec(x_vals1), label = r'Fit $(x < 32)$: $A + \frac{M}{1+ c_3\exp(-d_3 x)}, c_3 = \frac{M}{y_0-A}-1$')
    ax1.plot(x_vals2, fit_full_vec(x_vals2), label = r'Fit $(x \geq 32)$: $a_3(x-32)^2 + b_3(x-32) + A + \frac{M}{1+ c_3\exp(-d_3 \cdot 32)} - \Delta y$')

    d0 = generate_dictionary(fit_full, Ndatapoints = len(x))
    
    text = nice_string_output(d0, extra_spacing=2, decimals=2)
    add_text_to_ax(0.05, 0.75, text, ax1, fontsize=13)
    fig1.tight_layout()
    ax1.legend(fontsize = 13)
    plt.show()


def P5_2():
    residuals = np.loadtxt('data_TimingResiduals.txt')
    Npoints = len(residuals)
    print("We are given ", Npoints, "residuals")
    ### Find typical timing uncertainty. Is the mean consistent with 0 (one sample test)

    # Typical uncertainty is std of sample, so
    mean, std, sem = calc_mean_std_sem(residuals)
    print("Typical timining uncertainty, mean and sem: ", std, mean, sem)
    print("since (mean - 0)/ sem = ", mean/sem, " it is clealy consistent with 0")

    ### Based on sample size, are some residuals suspicious? Quantify
    mask1 = (np.abs(residuals - mean) > 2 *std)
    mask2 = (np.abs(residuals - mean) > 3 * std)
    mask3 = (np.abs(residuals - mean) > 3.5 * std)

    print("no. of residuals further than 2, 3, 3.5 std from mean: ", len(residuals[mask1]), len(residuals[mask2]), len(residuals[mask3]))

    print(f"Probability that any one point of a Gaussian distribution is further away than 3, 3.5 sigma given {Npoints} points: ", \
        (1 - (1 - 2 * stats.norm.sf(3)) ** Npoints),   (1 - (1 - 2 * stats.norm.sf(3.5)) ** Npoints))
    
    residuals_far = residuals[mask3]
    p_local = 2 * stats.norm.sf(residuals_far / std)
    p_global = 1 - (1 - p_local) ** Npoints
    arr = np.block([[residuals_far / std], [p_global]]).T
    print("ASSUMING GAUSSIAN ERRORS")
    print("Residuals more than 3.5 sigma away from mean in units of std and their global probability: \n", arr)
    print("We reject 3 points at 5% significance")
    print(residuals_far)
    cutoff_indices = np.argwhere(np.abs(residuals) / std < 5).flatten()
    residuals = residuals[cutoff_indices]
 
    ### Fit distribution with Gaussian

    fig, ax = plt.subplots()
    range = (np.min(residuals), np.max(residuals))
    bins = 80

    def gaussian_binned(x, N, mean, std):
        bin_width = (range[1] - range[0]) /  bins
        return N * bin_width * 1 / (np.sqrt(2 * np.pi) * std) * np.exp(-0.5 * (x-mean) ** 2 / std ** 2)

    def gaussian_LH(x, N, mean, std):
        return  N * 1 / (np.sqrt(2 * np.pi) * std) * np.exp(-0.5 * (x-mean) ** 2 / std ** 2)


    counts, edges, _ = ax.hist(residuals, bins = bins, range = range, histtype = 'stepfilled', lw = 2, alpha = 0.5, label = 'Histogram of time residuals')
    
    x_vals = 0.5 * (edges[:-1] + edges[1:])
    errors = np.sqrt(counts)
    mask = (counts > 0)
    N_non_empty = len(mask)

    ax.set(xlabel = 'Time residual (s)', ylabel = 'Count')
    ax.errorbar(x_vals, counts, errors, fmt = 'k.', elinewidth=1, capsize=1, capthick=1)

    parameter_guesses = np.array([1500, 0, 0.2])
    fit_chi2 = do_chi2_fit(gaussian_binned, x_vals[mask], counts[mask], errors[mask], parameter_guesses)
    fit_LH = do_LH_fit(gaussian_LH, residuals , fit_chi2.values, bound = range, unbinned = True, extended = True)

    if 1:
        fmax = gaussian_LH(fit_LH.values['mean'], *fit_LH.values)
        Nsimulations = 100
        LL_values, p_val = evaluate_likelihood_fit(gaussian_LH, fmax = fmax, parameter_val_arr = fit_LH.values, \
            log_likelihood_val = fit_LH.fval, bounds = range, Ndatapoints = len(residuals), Nsimulations = Nsimulations)

        plot_likelihood_fits(LL_values, p_val, log_likelihood_val = fit_LH.fval, Nsimulations = Nsimulations)
  
    x_fit = np.linspace(range[0], range[1], 1000)
    fit_vals_chi2 = gaussian_binned(x_fit, *fit_chi2.values[:])
    fit_vals_LH = gaussian_binned(x_fit, *fit_LH.values[:])
    ax.plot(x_fit, fit_vals_LH, label = 'LH Gaussian fit')
    ax.plot(x_fit, fit_vals_chi2, label = r'Chi2 Gaussian fit')
    
    Ndof, chi2, p = get_statistics_from_fit(fit_chi2, N_non_empty)
    d = {"Gaussian (chi2) fit:": " ","Ndof": Ndof, "chi2": chi2, 'Prop': p}
    #d = generate_dictionary(fit_chi2, Ndatapoints = N_non_empty, chi2_suffix = 'chi2')

    ### Fit with other PDFs. Compare to Gaussian fit.

    def double_gaussian_binned(x, N1, N2, mean1, mean2, std1, std2):
        bin_width = (range[1] - range[0]) /  bins
        val1 = N1 * bin_width * 1 / (np.sqrt(2 * np.pi) * std1) * np.exp(-0.5 * (x-mean1) ** 2 / std1 ** 2)
        val2 = N2 * bin_width * 1 / (np.sqrt(2 * np.pi) * std2) * np.exp(-0.5 * (x-mean2) ** 2 / std2 ** 2)
        return val1 + val2

    def t_density(x, N, mean, scale, df):
        bin_width = (range[1] - range[0]) /  bins
        val = bin_width * N * stats.t.pdf(x, loc = mean, scale = scale, df = df)
        return val

    parameter_guesses = np.array([500, 1000, 0, 0, 0.05, 0.2])
    fit_dbg = do_chi2_fit(double_gaussian_binned, x_vals[mask], counts[mask], errors[mask], parameter_guesses)
    ax.plot(x_vals, double_gaussian_binned(x_vals, *fit_dbg.values[:]), label = 'Double Gaussian fit')

    Ndof, chi2, p = get_statistics_from_fit(fit_dbg, N_non_empty)
    d0 = {"2x Gaussian fit:": " ","Ndof  ": Ndof, "chi2  ": chi2, 'Prop  ': p}
    d.update(d0)

    if 1:
        def double_gaussian_LH(x, N1, N2, mean1, mean2, std1, std2):
            val1 = N1 * 1 / (np.sqrt(2 * np.pi) * std1) * np.exp(-0.5 * (x-mean1) ** 2 / std1 ** 2)
            val2 = N2 * 1 / (np.sqrt(2 * np.pi) * std2) * np.exp(-0.5 * (x-mean2) ** 2 / std2 ** 2)
            return val1 + val2

        
        fit_LH = do_LH_fit(double_gaussian_LH, residuals, fit_dbg.values, bound = range)

        fmax = double_gaussian_LH(0, *fit_LH.values[:])
        Nsimulations = 500

        LL_values, p_val = evaluate_likelihood_fit(double_gaussian_LH, fmax = fmax, \
            parameter_val_arr = fit_LH.values, \
            log_likelihood_val = fit_LH.fval, bounds = range, Ndatapoints = len(residuals), Nsimulations = Nsimulations)

        plot_likelihood_fits(LL_values, p_val, log_likelihood_val = fit_LH.fval, Nsimulations = Nsimulations)
    
        ax.plot(x_vals, double_gaussian_binned(x_vals, *fit_LH.values[:]), label = '2xG LH')


    parameter_guesses = np.array([1500, 0.005, 0.01, 50])
    fit_t = do_chi2_fit(t_density, x_vals[mask], counts[mask], errors[mask], parameter_guesses)
    ax.plot(x_vals, t_density(x_vals, *fit_t.values[:]), label = 't distribution fit')

    Ndof, chi2, p = get_statistics_from_fit(fit_t, N_non_empty)
    d0 = {"t dist. fit:": " ","Ndof ": Ndof, "chi2 ": chi2, 'Prop ': p}
    d.update(d0)

    text = nice_string_output(d, extra_spacing=0, decimals=2)
    add_text_to_ax(0.05, 0.9, text, ax, fontsize=14)

    ax.legend()
    fig.tight_layout()
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
rcParams['figure.figsize'] = (6,6)
rcParams['axes.prop_cycle'] = cycler(color = ['teal', 'navy', 'coral', 'plum', 'purple', 'lightblue', 'olivedrab', 'black', 'red', 'cyan', 'yellow', 'khaki'])
np.set_printoptions(precision = 5, suppress=1e-10)

## Set parameters and which problems to run
p1_1, p1_2, p1_3  = False, True, False
p2_1, p2_2, p3_1, p3_2 = False, False, False, False
p4_1, p4_2, p5_1, p5_2 =  False, False, False, True



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


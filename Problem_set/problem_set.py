# Autoher: Simon Guldager Andersen
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
from statsmodels.sandbox.stats.runs import runstest_1samp, runstest_2samp

sys.path.append('Appstat2022\\External_Functions')
from ExternalFunctions import Chi2Regression, BinnedLH, UnbinnedLH
from ExternalFunctions import nice_string_output, add_text_to_ax    # Useful functions to print fit results on figure


## Change directory to current one
os.chdir('AppStat2022\\Problem_set')


### FUNCTIONS -------------------------------------------------------------------------------------------------------------------------------

def covariance(x,y):
    return np.mean(x * y) - x.mean() * y.mean()

def prop_err(dzdx, dzdy, x, y, dx, dy, correlation = 0):
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

def evaluate_likelihood_fit (fit_function, fmax, parameter_val_arr, log_likelihood_val, bounds, Ndatapoints, Nsimulations, Nbins = 0, unbinned = True):
    """
    fit_function is assumed to have the form f(x, *parameters), with x taking values in bounds
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
        fig0.legend( fontsize = 16, bbox_to_anchor = (0.25,0.65,0.2,0.2))
        fig0.tight_layout()
        return None

def runstest_homebrew(residuals):
   
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

def gaussian_binned(x, N, mean, std):
    bin_width = (range[1] - range[0]) /  bins
    return N * bin_width * 1 / (np.sqrt(2 * np.pi) * std) * np.exp(-0.5 * (x-mean) ** 2 / std ** 2)

def double_gaussian_binned(x, N1, N2, mean1, mean2, std1, std2):
    bin_width = (range[1] - range[0]) /  bins
    val1 = N1 * bin_width * 1 / (np.sqrt(2 * np.pi) * std1) * np.exp(-0.5 * (x-mean1) ** 2 / std1 ** 2)
    val2 = N2 * bin_width * 1 / (np.sqrt(2 * np.pi) * std2) * np.exp(-0.5 * (x-mean2) ** 2 / std2 ** 2)
    return val1 + val2
        
# Calculate ROC curve from two histograms (hist1 is signal, hist2 is background):
def calc_ROC(hist1, hist2) :

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
    
        # Initialize empty arrays for the True Positive Rate (TPR) and the False Positive Rate (FPR):
        TPR = np.zeros_like(y_sig) # True positive rate (sensitivity)
        FPR = np.zeros_like(y_sig) # False positive rate ()
        
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


### MAIN ------------------------------------------------------------------------------------------------------------------------------------

# Functions answering each problem

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





## Is the prob estimation ok?
def P2_2():
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

## What is meant exactly by determine a within 1 %???
def P3():
    ## Consider distribution f = C * x**3 * sin(pi * x) from x in [0,1]

    ## Determine normalization constant numerically
    f = lambda x: x ** 3 * np.sin (np.pi * x)
    area, err_area = integrate.quad(f, 0, 1)
    print("area: ", area, "error on area: ", err_area)
    C = 1 / area
    dC = np.sqrt((- 1 / area ** 2 ) ** 2 * err_area ** 2 )
    print(f'Value of C: {C:.14},  Uncertainty on C: {dC:.6}')

    # Find max value of f
    def f_norm(x):
        if x < 0 or x > 1:
            return 0
        else:
            return C * x ** 3 * np.sin (np.pi * x)

    f = lambda x: C * x ** 3 * np.sin (np.pi * x)
    fmax = - 1 * optimize.minimize(lambda x: - f_norm(x), 0.5)['fun']
  

    ## Fit a histogram with values from f and determine how many measurements / values of x you need
    # in an experiment to determine the value of a within 1%

    xmin, xmax = 0, 1
    N_points = 3000
    bounds = (xmin, xmax)
    x_accepted, _, _ = rejection_sampling_uniform(f, fmax, bounds = bounds, Npoints = N_points, verbose = True)

    bins = 40
    fig, ax = plt.subplots(figsize = (8,3))
    counts, edges, _ = ax.hist(x_accepted, bins=bins, range=(xmin, xmax), color = 'red', histtype='step', label='histogram', linewidth = 2)
    ax.set(xlim=(xmin, xmax+0.05))
    ax.set_title('Values sampled from the distribution y = C * x^3 * sin (pi x) and corresponding fit', fontsize = 20)
    ax.set_xlabel( xlabel="x value", fontsize = 18)
    ax.set_ylabel( ylabel="y value", fontsize = 18)
    x_vals = 0.5 * ( edges[:-1] + edges[1:])
    y_vals = counts
    y_err = np.sqrt(counts)
    mask = (y_vals > 0)
    N_non_empty = len(mask)

    ax.errorbar(x_vals, counts, y_err, fmt = 'k.', elinewidth=1.5, capsize=1.5, capthick=1)
    # Fit
    def fit_func(x, a):
        bin_width = (xmax - xmin) / bins
        scaling = N_points * bin_width
        return scaling * C * x ** a * np.sin (np.pi * x)


    chi2_object = Chi2Regression(fit_func, x_vals[mask], y_vals[mask], y_err[mask])
    chi2_object.errordef = 1
    fit = iminuit.Minuit(chi2_object, a = 3)
    fit.migrad()

    # plot fit
    x_range = np.linspace(xmin, xmax, 1000)

    fit_vals =  fit_func(x_range, *fit.values[:])

    ax.plot(x_range, fit_vals, 'k-', linewidth = 2)

    # Get statistics
    Ndof = len(y_vals[mask]) - len(fit.values[:])
    chi2 = fit.fval
    prop = stats.chi2.sf(chi2, Ndof)
    rel_error = np.abs((fit.values['a'] - 3)) / 3
    d = {"Entries": N_points, "Fit function": 'y = C * x^a * sin(pi x)', "fit param a": [fit.values['a'],fit.errors['a']], "rel error on a": rel_error,
            "Ndof": Ndof, "Chi squared": chi2, "Prop": prop}

    # Plot figure text
    text = nice_string_output(d, extra_spacing=2, decimals=3)
    add_text_to_ax(0.05, 0.90, text, ax, fontsize=16)

    fig.tight_layout()
    plt.show()

def P4_1():
    ## Load data, 1st column records the dom hand (0 = left hand, 1 = right hand), 2nd column the grip strength of dominant hand
    # 3rd column the grip strength of the non-dominant hand
    dom_hand, dom_strength, non_dom_strength = np.loadtxt('data_GripStrength.csv', skiprows = 1, unpack = True, delimiter=',')
    assert (dom_hand.shape == dom_strength.shape == non_dom_strength.shape)

    ## What fraction are right handed (dom_hand = 1)?
    N_right_handed = len(np.argwhere(dom_hand == 1).flatten())
    print(f'Fraction of the 84 people who are right-handed: ', N_right_handed / len(dom_hand))

    ## What is the mean and std of the dominant and non-dominant grip-strentghs?
    dom_mean = dom_strength.mean()
    non_dom_mean = non_dom_strength.mean()
    dom_std = dom_strength.std(ddof = 1)
    non_dom_std = non_dom_strength.std(ddof = 1)
    dom_sem = dom_std / np.sqrt(len(dom_strength))
    non_dom_sem = non_dom_std / np.sqrt(len(non_dom_strength))

    print('Mean, std and EMS for dominant hand grip strength: ', dom_mean, " ", dom_std, " ", dom_sem)
    print('Mean, std and EMS for non-dominant hand grip strength: ', non_dom_mean, " ", non_dom_std, " ", non_dom_sem)
    print()

    plt.hist(dom_strength, histtype = 'step', color = 'blue')
    plt.hist(non_dom_strength, histtype = 'step', color = 'red')
    plt.show(block=False)

    
    z_val = (dom_mean - non_dom_mean) / np.sqrt(dom_std ** 2 / len(dom_strength) + non_dom_std ** 2 / len(non_dom_strength))
    p_val = stats.t.sf(z_val, df = 2 * len(dom_strength) - 2) # z-val and t-val are identical

    # alternatively::: 
    # t_val, p_val = stats.ttest_ind(dom_strength, non_dom_strength, equal_var = False, alternative = 'greater')
    print("two sample one-tailed t_test t-value and p-value: ", z_val, " ",p_val)
    print("two sample one-tailed z-test z-value and p_value: ", z_val, stats.norm.sf(z_val))

   
    print("The null-hypothesis that the two distributions have identical means cannot be rejected to 5 percent significance")

    # Find mean and std of the individual differences in grip strenghts?
    strength_diff = dom_strength - non_dom_strength
    strength_diff_mean = strength_diff.mean()
    strength_diff_std = strength_diff.std(ddof = 1)
    strength_diff_ems = strength_diff_std / np.sqrt(len(strength_diff))
    print("Mean, std and EMS for individual grip strength difference: ", strength_diff_mean, " ", strength_diff_std, " ", strength_diff_ems)

    # Assume that there is no statistical difference in grip strength between hands, i.e. that the mean difference
    # belong to a distribution of mean = 0. and uncertainty = strength_diff_std / sqrt(N)

    z_val = (strength_diff_mean) / (strength_diff_std / np.sqrt(len(strength_diff)))
    reference = np.zeros_like(strength_diff)
    t_val, p_val = stats.ttest_ind(strength_diff, reference, alternative = 'greater')

    print("two sample one-tailed t_test t-value and p-value: ", t_val, " ",p_val)
    print("two sample one-tailed z-test z-value and p_value: ", z_val, stats.norm.cdf(-z_val))


    chi2 = np.sum(strength_diff ** 2 / strength_diff_std ** 2)

    print("chi squared and p_val ", chi2," ", stats.chi2.sf(chi2, len(strength_diff) - 1))

    plt.hist(strength_diff, histtype = 'stepfilled', linewidth = 2, color = 'blue')
    plt.show()

# Should you include errors from fit when doing sample purity curves? And how?
def P4_2():
    ## Load data
    size, intensity = np.loadtxt('data_MoleculeTypes.csv', skiprows = 1, unpack = True, delimiter = ',')
    assert (size.shape == intensity.shape)

    size_plot = True # If true, all plots, fits and ROC curves are made for the intensity data

    # Does the molecule size follow a Gaussian distribution? How about when only considering I > 0.5?
    # Fit the distribution with a Gaussian, then the I>0.5 distribution with a Gaussian, then with 2 gaussians
  
    if 0:
        xbins, ybins = 44, 50
        hist_2d,xEdges,yEdges,im = plt.hist2d(size, intensity, cmap = 'Blues', cmin = 0, bins=(xbins,ybins), range = ((0,70),(0,1)))
        plt.xlabel('Molecule size (microns)', fontsize = 18)
        plt.ylabel('Intensity (relative)', fontsize = 18)
        plt.title('Joint histogram distribution of intensity size', fontsize = 18)
        x_centers = 0.5 * (xEdges[:-1] + xEdges[1:])
        y_centers = 0.5 * (yEdges[:-1] + yEdges[1:])
        a = - np.abs ((yEdges[1] - yEdges[0]) / ( xEdges[1] - xEdges[0]))
        b = 0.92
        y_vals = a * x_centers + b
        plt.plot(x_centers, y_vals, 'k--')

        sigx1, mux1 = 7.4, 28
        sigx2, mux2 = 8.9, 45.7
        sigi1, mui1 = 0.1, 0.4
        sigi2, mui2 = 0.1, 0.7 

        def gauss2d(x, y, sigma_x, mu_x, sigma_y, mu_y, A, v):

            a = np.cos(v)**2 / (2 * sigma_x**2) + np.sin(v)**2 /(2 * sigma_y **2)
            b = - np. sin(2 * v) / (4 * sigma_x**2) + np.sin(2*v) / (4 * sigma_y ** 2)
            c = np.sin(v)**2 / (2 * sigma_x ** 2) + np.cos(v) ** 2 / (2 * sigma_y ** 2)

            val = A * np.exp(-(a * (x - mu_x)**2 + 2 * b * (x - mu_x) * (y- mu_y) + c * (y - mu_y) ** 2))
            return val

    

        plt.colorbar(im)
    
        theta = - np.arctan(a)
        theta = 0
        X, Y = np.meshgrid(x_centers, 1 - y_centers)
        X, Y = np.meshgrid(np.linspace(0,70,1000), 1 - np.linspace(0,1,1000))

        y_vals2 = gauss2d(X, Y, sigx1, mux1, sigi1, mui1, 200, theta)
        y_vals3 = gauss2d(X, Y, sigx2, mux2, sigi2, mui2, 200, theta)

        plt.contour(y_vals3)

        print(hist_2d.shape)
        hist_arr = np.flip(hist_2d, axis = 1).T

        y_start = ybins - int(92/2  )

        area = 0
        for i in np.arange(xbins):
            area += np.sum(hist_arr[y_start + i:,i])
            print(area)
            print("hist", hist_arr[y_start + i:,i])
        print(area, hist_arr.sum())



        plt.show()
    
    bins = int(len(size) / 50)
    range = (np.min(size), np.max(size))

    def gaussian_binned(x, N, mean, std):
        bin_width = (range[1] - range[0]) /  bins

        return N * bin_width * 1 / (np.sqrt(2 * np.pi) * std) * np.exp(-0.5 * (x-mean) ** 2 / std ** 2)

    def gaussian_LH(x, N, mean, std):
        return  N * 1 / (np.sqrt(2 * np.pi) * std) * np.exp(-0.5 * (x-mean) ** 2 / std ** 2)


    def double_gaussian_binned(x, N1, N2, mean1, mean2, std1, std2):
        bin_width = (range[1] - range[0]) /  bins
 
        val1 = N1 * bin_width * 1 / (np.sqrt(2 * np.pi) * std1) * np.exp(-0.5 * (x-mean1) ** 2 / std1 ** 2)
        val2 = N2 * bin_width * 1 / (np.sqrt(2 * np.pi) * std2) * np.exp(-0.5 * (x-mean2) ** 2 / std2 ** 2)
        return val1 + val2

    
    def double_gaussian_LH(x, N1, N2, mean1, mean2, std1, std2):
    
        val1 = N1 * 1 / (np.sqrt(2 * np.pi) * std1) * np.exp(-0.5 * (x-mean1) ** 2 / std1 ** 2)
        val2 = N2 * 1 / (np.sqrt(2 * np.pi) * std2) * np.exp(-0.5 * (x-mean2) ** 2 / std2 ** 2)
        return val1 + val2

    
    # define fitting functions
    if size_plot:
        fig, ax = plt.subplots(nrows = 2, ncols = 2, figsize = (8,12))
        ax = ax.flatten()

     

        # extract indices where I > 0.5
        index_50 = np.argwhere(intensity > 0.50).flatten()
        size_50 = size[index_50]

        range = (np.min(size), np.max(size))
        val_list = [size, size_50, size]
        func_list = [gaussian_binned, gaussian_binned, double_gaussian_binned]
        func_list_LH = [gaussian_LH, gaussian_LH, double_gaussian_LH]
        parameter_guesses = [np.array([900, 50, 3]),np.array([700, 55, 3]), np.array([500, 300, 55, 25, 3, 3])]
        name_list = ['Size distribution','Size distribution for I > 0.5' ,'Size distribution']


        for i, (fit_func, fit_func_LH) in enumerate(zip(func_list, func_list_LH)):
        
            counts, edges, _ = ax[i].hist(val_list[i], bins = bins, range = range, histtype = 'step', linewidth = 2)

            
            ax[i].set_xlabel(xlabel = 'Size (microns)', size = 16)
            ax[i].set_ylabel(ylabel = 'No. of molecules', size = 16)
            ax[i].set_title(f'{name_list[i]}', size = 18)

            x_vals = 0.5 * ( edges[:-1] + edges[1:])
            y_vals = counts
            y_err = np.sqrt(counts)
            mask = (y_vals > 0)
            N_non_empty = len(mask)

            # ax[i].errorbar(x_vals, counts, y_err, fmt = 'k.', elinewidth=1.5, capsize=1.5, capthick=1)

            if 0:
                binned_likelihood = BinnedLH(lambda y: fit_func(y, chi2 = False), 
                                        val_list[i],
                                        bins=bins, 
                                        bound=(range[0], range[1]),
                                        extended=True)

                binned_likelihood.errordef = iminuit.Minuit.LIKELIHOOD 


            chi2_object = Chi2Regression(fit_func, x_vals[mask], y_vals[mask], y_err[mask])
            chi2_object.errordef = 1
            fit = iminuit.Minuit(chi2_object, *parameter_guesses[i])

            ULH_object = UnbinnedLH(fit_func_LH, val_list[i], bound = (range[0], range[1]), extended = True)
            ULH_object.errordef = iminuit.Minuit.LIKELIHOOD

            fit_ULH = iminuit.Minuit(ULH_object, *fit.values[:])
            print(fit.migrad())
            print(fit_ULH.migrad())
            # plot fit
            x_range = np.linspace(range[0], range[1], 1000)


            # Get statistics
            Ndof = len(y_vals[mask]) - len(parameter_guesses[i])
            chi2 = fit.fval
            prop = stats.chi2.sf(chi2, Ndof)
            
            if i<=1:
                d = {"Entries": len(val_list[i]), "fitted N": [fit.values['N'],fit.errors['N']],  "fitted mean": [fit.values['mean'],fit.errors['mean']], \
                    "fitted std": [fit.values['std'],fit.errors['std']],
                    "Ndof": Ndof, "Chi squared": chi2, "Prop": prop}
                    
                fit_vals =  fit_func(x_range, *fit.values[:])
                fit_vals_LH = fit_func(x_range, *fit_ULH.values[:])
                ax[i].plot(x_range, fit_vals, 'k-', linewidth = 2, label = 'chi2 Gaussian fit')
                ax[i].plot(x_range, fit_vals_LH, 'g-', lw = 2, label = 'LH Gaussian fit')
                
            else:
                d = d = {"Entries": len(val_list[i]), "fitted N1": [fit.values['N1'],fit.errors['N1']], "fitted N2": [fit.values['N2'],fit.errors['N2']], \
                        "fitted mean1": [fit.values['mean1'],fit.errors['mean1']],  "fitted mean2": [fit.values['mean2'],fit.errors['mean2']],\
                    "fitted std1": [fit.values['std1'],fit.errors['std1']], "fitted std2": [fit.values['std2'],fit.errors['std2']], \
                    "Ndof": Ndof, "Chi squared": chi2, "Prop": prop} 
                single_gauss = gaussian_binned
                fit_vals1 =  single_gauss(x_range, fit.values['N1'], fit.values['mean1'], fit.values['std1'])
                fit_vals2 =  single_gauss(x_range, fit.values['N2'], fit.values['mean2'], fit.values['std2'])
                fit_vals3 = double_gaussian_binned(x_range, *fit.values[:])
                fit_vals1LH =  single_gauss(x_range, fit_ULH.values['N1'], fit_ULH.values['mean1'], fit_ULH.values['std1'])
                fit_vals2LH =  single_gauss(x_range, fit_ULH.values['N2'], fit_ULH.values['mean2'], fit_ULH.values['std2'])
                fit_vals3LH = double_gaussian_binned(x_range, *fit_ULH.values[:])
                ax[i].plot(x_range, fit_vals1, 'k-', linewidth = 2, label = 'chi2 Gaussian fit 1')
                ax[i].plot(x_range, fit_vals2, 'r-', linewidth = 2, label = 'chi2 Gaussian fit 2')
                ax[i].plot(x_range, fit_vals3, 'm--', linewidth = 2, label = 'chi2 Double Gaussian fit')

                ax[i].plot(x_range, fit_vals1LH, '-', linewidth = 2, label = 'LH Gaussian fit 1')
                ax[i].plot(x_range, fit_vals2LH, '-', linewidth = 2, label = 'LH Gaussian fit 2')
                ax[i].plot(x_range, fit_vals3LH, '--', linewidth = 2, label = 'LH Double Gaussian fit')


            ax[i].legend(loc='best', fontsize = 10)
            fit_vals = fit.values[:]
            # Plot figure text
            text = nice_string_output(d, extra_spacing=2, decimals=1)
            add_text_to_ax(0.05, 0.95, text, ax[i], fontsize=11)


        fig.tight_layout()


        ## Calc. p-value for Gaussian double fit unbinned likelihood
        fmax = np.max(counts)
       
        Nsimulations = 50
        LL_values, p_val = evaluate_likelihood_fit(double_gaussian_LH, fmax, fit_ULH.values[:], fit_ULH.fval, \
             bounds = (range[0], range[1]), Ndatapoints = int(counts.sum()), Nsimulations = Nsimulations, unbinned = True)

        plot_likelihood_fits(LL_values, p_val, fit_ULH.fval, Nsimulations)
       

        ## Calculate sample purity and ROC-curve (assuming that the to Gaussian fits are good)
        # To be able to use the calc-ROC function, which takes right histogram on the right as a signal, we
        # swap the two histograms by letting mean --> -mean
        fig2, ax2 = plt.subplots(ncols = 2,figsize = (12,6))
        ax2 = ax2.flatten()

        bins = int(len(size)/2)
        # Range reflected around 0
        range = (- np.max(size), - np.min(size))
        range_plot = (np.min(size), np.max(size))
        bin_width = (range[1]- range[0]) / bins
        edges = np.arange(range[0], range[1], bin_width)
        bin_centers = 0.5 * (edges[:-1] + edges[1:])
        # Define entries for histogram 1, and scale entries for histogram 2 accordingly
        scale = 1000
        N_fit1 = scale * fit.values['N1']
        N_fit2 = fit.values['N2'] / fit.values['N1'] * N_fit1

        # generate values according to each Gaussian, but letting mean --> -mean
        gauss_vals_fit1 =   stats.norm.rvs(loc = -fit.values['mean1'], scale = fit.values['std1'], size = int(N_fit1))
        gauss_vals_fit2 =  stats.norm.rvs(loc = -fit.values['mean2'], scale = fit.values['std2'], size = int(N_fit2))

        gauss_vals_plot1 =   stats.norm.rvs(loc = fit.values['mean1'], scale = fit.values['std1'], size = int(N_fit1))
        gauss_vals_plot2 =  stats.norm.rvs(loc = fit.values['mean2'], scale = fit.values['std2'], size = int(N_fit2))

        hist1 = plt.hist(gauss_vals_fit1, bins = bins, range = range, histtype = 'step', color = 'red')
        hist2 = plt.hist(gauss_vals_fit2, bins = bins, range = range, histtype = 'step', color = 'blue')

        ax2[1].hist(gauss_vals_plot1, bins = bins, range = range_plot, histtype = 'step', color = 'red', label = 'Background')
        ax2[1].hist(gauss_vals_plot2, bins = bins, range = range_plot, histtype = 'step', color = 'blue', label = 'Signal')
        ax2[1].set_xlim((0,np.max(size)+5))
        ax2[1].set_xlabel(xlabel = 'Molecule size (microns)', fontsize = 18)
        ax2[1].set_ylabel(ylabel = 'Count', fontsize = 18)
        ax2[1].set_title(label = 'Histograms generated from Gaussian fits', fontsize = 18)

        ax2[1].legend(loc = 'upper left', fontsize = 16)
        y_sig, x_sig_edges, _ = hist2
        y_bkg, x_bkg_edges, _ = hist1

        ROC, sample_pure = False, True

        if ROC:
            FPR, TPR,_ = calc_ROC(hist2, hist1)
            index90 = np.argmin(np.abs(TPR-0.9))
            ax2[0].plot(FPR, TPR, 'm-', linewidth = 2, label = 'ROC-curve')
            ax2[0].plot([FPR[index90],FPR[index90]],[0,TPR[index90]], 'k-')
            ax2[0].plot([-0.1,FPR[index90]],[TPR[index90],TPR[index90]], 'k-')
            ax2[0].plot(FPR[index90],TPR[index90], 'cx', markersize = '14')
            ax2[0].set_xlabel('False positive rate', fontsize = 18)
            ax2[0].set_ylabel('True positive rate', fontsize = 18)
            ax2[0].set_title('ROC curve for the new (small) molecule', fontsize = 18)
            ax2[0].set_xlim((-0.05,1.05))
        elif sample_pure:
            #_, edges, _ = plt.hist(gauss_vals_fit1, bins = bins, range = range, histtype = 'step', color = 'red')
            x_vals = - bin_centers

            SP = calc_sample_purity(hist2,hist1)
            index90 = np.argmin(np.abs(SP-0.9))
            ax2[0].plot(np.flip(x_vals), np.flip(SP), 'm-', linewidth = 2)
            ax2[0].plot([x_vals[index90],x_vals[index90]],[0,SP[index90]], 'k-')
            ax2[0].plot([0,x_vals[index90]],[SP[index90],SP[index90]], 'k-')
            ax2[0].plot(x_vals[index90],SP[index90], 'cx', markersize = '14')
            ax2[0].set_xlabel('Molecule size (microns) ', fontsize = 18)
            ax2[0].set_ylim((0.2,1.05))
            ax2[0].set_xlim((0,1.05))
            ax2[0].set_ylabel('Sample purity', fontsize = 18)
            ax2[0].set_title('Sample purity (small molecule) against size cutoff', fontsize = 18)
            ax2[0].set_xticks(np.arange(0,80,10))

            print("Molecule size cutoff to acchieve 90% purity: ", x_vals[index90])
        ax2[0].legend(loc = 'best', fontsize = 16)
        
        # Calculate number of molecules in sample
        x_cutoff = x_vals[index90]
        mask = (x_vals < x_cutoff)

        # Rembember to scale back
        N_small = y_sig[mask].sum() / scale
        N_big = y_bkg[mask].sum() / scale
        dN_small = fit.errors['N2'] * N_small / (y_sig.sum() / scale)
        dN_big = fit.errors['N1'] * N_big / (y_bkg.sum() / scale)

        cut = (size <= x_cutoff)
        N_molecules_remaining = len(size[cut])
    

        print(N_small/(N_small+N_big))
        print("Number of small molecules in sample (fit est): ", N_small, "\u00B1", np.round(dN_small,1))
        print("Number of big molecules in sample (fit est): ", N_big, "\u00B1", np.round(dN_big,1))
        print("Total number of molecules in sample (fit est): ", N_small + N_big, "\u00B1", np.round(np.sqrt(dN_small**2 + dN_big**2),1))
        print("Actual no. of molecules remaining in sample after cutoff: ", N_molecules_remaining)
        fig2.tight_layout()
        plt.show()
    else:
            fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (14,6))
            ax = ax.flatten()

            bins = int(len(size) / 50)
            range = (np.min(intensity), np.max(intensity))
            
            func_list = [gaussian_binned, double_gaussian_binned]
            range = (np.min(intensity), np.max(intensity))
            val_list = [intensity, intensity]
            parameter_guesses = [np.array([900, 0.7, 0.1]), np.array([300, 500, 0.7, 0.2, 0.05, 0.05])]
            name_list = ['Intensity distribution', 'Intensity distribution']
           
        
            for i, fit_func in enumerate(func_list):
            
                counts, edges, _ = ax[i].hist(val_list[i], bins = bins, range = range, histtype = 'step', linewidth = 2)

                
                ax[i].set_xlabel(xlabel = 'Intensity (relative)', size = 16)
                ax[i].set_ylabel(ylabel = 'No. of molecules', size = 16)
                ax[i].set_title(f'{name_list[i]}', size = 18)
              #  ax[i].set_xlim=((0,1.1))

                x_vals = 0.5 * ( edges[:-1] + edges[1:])
                y_vals = counts
                y_err = np.sqrt(counts)
                mask = (y_vals > 0)
                N_non_empty = len(mask)

                # ax[i].errorbar(x_vals, counts, y_err, fmt = 'k.', elinewidth=1.5, capsize=1.5, capthick=1)

                binned_likelihood = BinnedLH(fit_func, 
                                        val_list[i],
                                        bins=bins, 
                                        bound=(range[0], range[1]),
                                        extended=True)

                binned_likelihood.errordef = iminuit.Minuit.LIKELIHOOD 
                chi2_object = Chi2Regression(fit_func, x_vals[mask], y_vals[mask], y_err[mask])
                chi2_object.errordef = 1
                fit = iminuit.Minuit(chi2_object, *parameter_guesses[i])
                #fit = iminuit.Minuit(binned_likelihood, *parameter_guesses[i])
                print(fit.migrad())

                # plot fit
                x_range = np.linspace(range[0], range[1], 1000)


                # Get statistics
                Ndof = len(y_vals[mask]) - len(parameter_guesses[i])
                chi2 = fit.fval
                prop = stats.chi2.sf(chi2, Ndof)
                
                if i==0:
                    d = {"Entries": len(val_list[i]), "fitted N": [fit.values['N'],fit.errors['N']],  "fitted mean": [fit.values['mean'],fit.errors['mean']], \
                        "fitted std": [fit.values['std'],fit.errors['std']],
                        "Ndof": Ndof, "Chi squared": chi2, "Prop": prop}
                        
                    fit_vals =  fit_func(x_range, *fit.values[:])
                    ax[i].plot(x_range, fit_vals, 'k-', linewidth = 2, label = 'Gaussian fit')
                    
                else:
                    d = d = {"Entries": len(val_list[i]), "fitted N1": [fit.values['N1'],fit.errors['N1']], "fitted N2": [fit.values['N2'],fit.errors['N2']], \
                            "fitted mean1": [fit.values['mean1'],fit.errors['mean1']],  "fitted mean2": [fit.values['mean2'],fit.errors['mean2']],\
                        "fitted std1": [fit.values['std1'],fit.errors['std1']], "fitted std2": [fit.values['std2'],fit.errors['std2']], \
                        "Ndof": Ndof, "Chi squared": chi2, "Prop": prop} 
                    single_gauss = gaussian_binned
                    fit_vals1 =  single_gauss(x_range, fit.values['N1'], fit.values['mean1'], fit.values['std1'])
                    fit_vals2 =  single_gauss(x_range, fit.values['N2'], fit.values['mean2'], fit.values['std2'])
                    fit_vals3 = double_gaussian_binned(x_range, *fit.values[:])
                    ax[i].plot(x_range, fit_vals1, 'k-', linewidth = 2, label = 'Gaussian fit 1')
                    ax[i].plot(x_range, fit_vals2, 'r-', linewidth = 2, label = 'Gaussian fit 2')
                    ax[i].plot(x_range, fit_vals3, 'm--', linewidth = 2, label = 'Double Gaussian fit')

                    ax[i].legend(loc='upper right', fontsize = 11)
                    fit_vals = fit.values[:]
                # Plot figure text
                text = nice_string_output(d, extra_spacing=2, decimals=1)
                add_text_to_ax(0.05, 0.95, text, ax[i], fontsize=11)


            fig.tight_layout()


            ## Calculate sample purity and ROC-curve (assuming that the to Gaussian fits are good)
            # To be able to use the calc-ROC function, which takes right histogram on the right as a signal, we
            # swap the two histograms by letting mean --> -mean
            fig2, ax2 = plt.subplots(ncols = 2,figsize = (12,6))
            ax2 = ax2.flatten()

            bins = int(len(intensity)/2)
            # Range reflected around 0
            range = (- np.max(intensity), - np.min(intensity))
            range_plot = (np.min(intensity), np.max(intensity))
            bin_width = (range[1]- range[0]) / bins
            edges = np.arange(range[0], range[1] + bin_width, bin_width)
            bin_centers = 0.5 * (edges[:-1] + edges[1:])
            # Define entries for histogram 1, and scale entries for histogram 2 accordingly
            scale = 1000
            N_fit1 = scale * fit.values['N1']
            N_fit2 = fit.values['N2'] / fit.values['N1'] * N_fit1

            # generate values according to each Gaussian, but letting mean --> -mean
            gauss_vals_fit1 =   stats.norm.rvs(loc = -fit.values['mean1'], scale = fit.values['std1'], size = int(N_fit1))
            gauss_vals_fit2 =  stats.norm.rvs(loc = -fit.values['mean2'], scale = fit.values['std2'], size = int(N_fit2))

            gauss_vals_plot1 =   stats.norm.rvs(loc = fit.values['mean1'], scale = fit.values['std1'], size = int(N_fit1))
            gauss_vals_plot2 =  stats.norm.rvs(loc = fit.values['mean2'], scale = fit.values['std2'], size = int(N_fit2))

            hist1 = plt.hist(gauss_vals_fit1, bins = bins, range = range, histtype = 'step', color = 'red')
            hist2 = plt.hist(gauss_vals_fit2, bins = bins, range = range, histtype = 'step', color = 'blue')

            ax2[1].hist(gauss_vals_plot1, bins = bins, range = range_plot, histtype = 'step', color = 'red', label = 'Background')
            ax2[1].hist(gauss_vals_plot2, bins = bins, range = range_plot, histtype = 'step', color = 'blue', label = 'Signal')
            ax2[1].set_xlim((0,np.max(intensity)+0.05))
            ax2[1].set_xlabel(xlabel = 'Intensity (relative)', fontsize = 18)
            ax2[1].set_ylabel(ylabel = 'Count', fontsize = 18)
            ax2[1].set_title(label = 'Histograms generated from Gaussian fits', fontsize = 18)

            ax2[1].legend(loc = 'upper left', fontsize = 16)
            y_sig, x_sig_edges, _ = hist2
            y_bkg, x_bkg_edges, _ = hist1

            ROC, sample_pure = False, True

            if ROC:
                FPR, TPR = calc_ROC(hist2, hist1)
                index90 = np.argmin(np.abs(TPR-0.9))
                ax2[0].plot(FPR, TPR, 'm-', linewidth = 2, label = 'ROC-curve')
                ax2[0].plot([FPR[index90],FPR[index90]],[0,TPR[index90]], 'k-')
                ax2[0].plot([-0.1,FPR[index90]],[TPR[index90],TPR[index90]], 'k-')
                ax2[0].plot(FPR[index90],TPR[index90], 'cx', markersize = '14')
                ax2[0].set_xlabel('False positive rate', fontsize = 18)
                ax2[0].set_ylabel('True positive rate', fontsize = 18)
                ax2[0].set_title('ROC curve for the new (small) molecule', fontsize = 18)
                ax2[0].set_xlim((-0.05,1.05))
            elif sample_pure:
                #_, edges, _ = plt.hist(gauss_vals_fit1, bins = bins, range = range, histtype = 'step', color = 'red')
                x_vals = - bin_centers

                SP = calc_sample_purity(hist2,hist1)
                index90 = np.argmin(np.abs(SP-0.9))
                ax2[0].plot(np.flip(x_vals), np.flip(SP), 'm-', linewidth = 2)
                ax2[0].plot([x_vals[index90],x_vals[index90]],[0,SP[index90]], 'k-')
                ax2[0].plot([0,x_vals[index90]],[SP[index90],SP[index90]], 'k-')
                ax2[0].plot(x_vals[index90],SP[index90], 'cx', markersize = '14')
                ax2[0].set_xlabel('Intensity (relative) ', fontsize = 18)
                ax2[0].set_ylim((0.2,1.05))
                ax2[0].set_xlim((0,1.05))
                ax2[0].set_ylabel('Sample purity', fontsize = 18)
                ax2[0].set_title('Sample purity (small molecule) against intensity cutoff', fontsize = 18)
                ax2[0].set_xticks(np.arange(0,1.1,0.1))

                print("Intensity cutoff to acchieve 90% purity: ", x_vals[index90])
            ax2[0].legend(loc = 'best', fontsize = 16)
            
            # Calculate number of molecules in sample
            x_cutoff = x_vals[index90]
            mask = (x_vals < x_cutoff)

            # Rembember to scale back
            N_small = y_sig[mask].sum() / scale
            N_big = y_bkg[mask].sum() / scale
            dN_small = fit.errors['N2'] * N_small / (y_sig.sum() / scale)
            dN_big = fit.errors['N1'] * N_big / (y_bkg.sum() / scale)

            cut = (intensity <= x_cutoff)
            N_molecules_remaining = len(intensity[cut])
        

            print(N_small/(N_small+N_big))
            print("Number of small molecules in sample (fit est): ", N_small, "\u00B1", np.round(dN_small,1))
            print("Number of big molecules in sample (fit est): ", N_big, "\u00B1", np.round(dN_big,1))
            print("Total number of molecules in sample (fit est): ", N_small + N_big, "\u00B1", np.round(np.sqrt(dN_small**2 + dN_big**2),1))
            print("Actual no. of molecules remaining in sample after cutoff: ", N_molecules_remaining)
            fig2.tight_layout()
            plt.show()
        

def P5_1():
    ## Load data [time in s and area in cm^2]
    time, area, area_err = np.loadtxt('data_AlgaeGrowth.csv', skiprows = 1, unpack = True, delimiter = ',')
    assert (time.shape == area.shape == area_err.shape)

    area_err *= 0.35
    # Plot a fit with a third degree polynomial. Is it good?
    def pol_third_order(x, a, b, c, d):
        return a * x ** 3 + b * x ** 2 + c * x + d

    def pol_oscillatory(x, a, b, c, d, A, phi): #, omega = np.pi):
        return a * x ** 3 + b * x ** 2 + c * x + d + A * np.cos(2 * np.pi * x + phi)

    fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (14,8))
    ax = ax.flatten()
    range = [np.min(time), np.max(time)]

    title_list = ['Algae growth over time', 'Algae growth over time']
    fit_list =  ['Fit y = at^3+bt^2+ct+d', 'Fit y = at^3+bt^2+ct+d+Acos(2pi*t+phi)']
    func_list = [pol_third_order, pol_oscillatory]
    parameter_guesses = [np.array([1, -10, 120, 1400]), np.array([1, -10, 120, 1400, 30, 4])]

    for i, fit_func in enumerate(func_list):
        ax[i].set_xlabel(xlabel = 'Time (days)', size = 18)
        ax[i].set_ylabel(ylabel = 'Area (cm^2)', size = 18)
        ax[i].set_title(f'{title_list[i]}', size = 20)

        ax[i].errorbar(time, area, area_err, fmt = 'r.', elinewidth=1, capsize=1, capthick=1)

        # Optimize parameters according to chi2 value
        chi2_object = Chi2Regression(fit_func, time, area, area_err)
        chi2_object.errordef = 1
        fit = iminuit.Minuit(chi2_object, *parameter_guesses[i])
        fit.migrad()

        # plot fit
        x_range = np.linspace(range[0], range[1], 1000)
        ax[i].plot(x_range, fit_func(x_range, *fit.values[:]), 'k-', linewidth = 2, label = f'{fit_list[i]}')
        ax[i].legend(loc = 'upper left', fontsize = '16')

        # do a runs test
        resids = area - fit_func(time, *fit.values[:])
        print("mean resids ", resids.mean())
        z_val, p_val = runstest_1samp(resids, cutoff = 0, correction = False)
        print(f'runs test z-statistics and p_value for fit {i+1}: ',  z_val,"  ", p_val)

        # homemade runs test
        z_val, p_val = runstest_homebrew(resids)
        print(f'homebrew runs test z-statistics and p_value for fit {i}: ',  z_val,"  ", p_val)

        # Get statistics
        Ndof = len(time) - len(fit.values[:])
        chi2 = fit.fval
        prop = stats.chi2.sf(chi2, Ndof)
      
        if i == 0:
            d = {"a (fit)": [fit.values['a'],fit.errors['a']],  "b (fit)": [fit.values['b'],fit.errors['b']], \
             "c (fit)": [fit.values['c'],fit.errors['c']], "d (fit)": [fit.values['d'],fit.errors['d']], \
                "Ndof": Ndof, "Chi squared": chi2, "Prop": prop}
        else:
            d = { "a (fit)": [fit.values['a'],fit.errors['a']],  "b (fit)": [fit.values['b'],fit.errors['b']], \
             "c (fit)": [fit.values['c'],fit.errors['c']], "d (fit)": [fit.values['d'],fit.errors['d']], \
                "A (fit)": [fit.values['A'],fit.errors['A']], \
                "phi (fit)": [fit.values['phi'],fit.errors['phi']], "Ndof": Ndof, "Chi squared": chi2, "Prop": prop}
            # "omega (fit)": [fit.values['omega'],fit.errors['omega']], 
        # Plot figure text
        text = nice_string_output(d, extra_spacing=2, decimals=2)
        add_text_to_ax(0.35, 0.35, text, ax[i], fontsize=13)

    fig.tight_layout()
    plt.show()

def P5_2():
    ## Load data [time in s and area in cm^2]
    wavelength, voltage = np.loadtxt('data_BohrHypothesis.csv', skiprows = 1, unpack = True, delimiter = ',')
    assert (wavelength.shape == voltage.shape)
    # Consider only wavelengths from 1200 to 2200 nm
    mask = ((wavelength > 1200) & (wavelength < 2200))
    wavelength = wavelength[mask]
    voltage = voltage[mask]
    xbins, ybins = 600, 400
    xrange = (1200,2200)
    yrange = (40,180)

    fig0, ax0 = plt.subplots(figsize = (12,8))
    counts,x_edges,y_edges,im = ax0.hist2d(wavelength, voltage, range =(xrange, yrange), bins = (xbins, ybins), cmin = 1)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    cutoff1 = 14

    # find average y_value for each x value
    rel_val_ind = np.argwhere(counts > cutoff1)
    unique_ind_x = np.unique(rel_val_ind[:,0])

    av_val_list = np.empty_like(unique_ind_x)

    for i in np.arange(len(unique_ind_x)):
        ind_i = unique_ind_x[i]
        sum = 0

        sub_matrix = np.argwhere(rel_val_ind == ind_i)
        y_sum = rel_val_ind[sub_matrix[:,0],1].sum()
        av_val_list[i] = y_sum / np.size(sub_matrix[:,0])


    wl1 = 1276.16
    wl2 = 1865.66

    # transform indices to x and y values (bins of centers)
    bin_width_x = x_centers[1] - x_centers[0]
    bin_width_y = y_centers[1] - y_centers[0]
    y_vals = yrange[0] + (yrange[1] - yrange[0])/ybins * av_val_list + 0.5 * bin_width_y
    x_vals = xrange[0] + (xrange[1] - xrange[0])/xbins * unique_ind_x + 0.5 * bin_width_x
    # mask to extract x and y values of 1st peak
    mask1 = (x_vals < 1500)
    y_vals1 = y_vals[mask1]
    y_vals2 = y_vals[~mask1]
    x_vals1 = x_vals[mask1]
    x_vals2 = x_vals[~mask1]

    plt.plot(x_vals, y_vals, 'r.', markersize = '6')

    def linfit1(x, a, b):
        return a * (x - wl1) + b

    def linfit2(x, a, b):
        return a * (x - wl2) + b

    def line_fit1(a,b):
        return np.sum(np.abs(y_vals1 - a *(x_vals1 - wl1) - b))

    def line_fit2(a,b):
        return np.sum(np.abs(y_vals2- a *(x_vals2 - wl2) - b))

    def line_fit_comb(a,b):
        return np.sum(np.abs(y_vals2- a *(x_vals2 - wl2) - b))
    # make linear fits


    fit1 = iminuit.Minuit(line_fit1, a = 10, b = 100)
    fit1.errordef = 1
    fit2 = iminuit.Minuit(line_fit2, a = 10, b = 100)
    fit2.errordef = 1
    print(fit1.migrad())
    print(fit2.migrad())
   
    lambda_trans2 = wavelength - 1/fit2.values['a'] * (voltage - fit2.values['b'])
    lambda_trans1 = wavelength - 1/fit1.values['a'] * (voltage - fit1.values['b'])

    #refit with residuals as uncertainties
    residuals1 = y_vals1 - linfit1(x_vals1, fit1.values['a'], fit1.values['b'])
    residuals2 = y_vals2 - linfit2(x_vals2, fit2.values['a'], fit2.values['b'])
    std1 = residuals1.std(ddof = 1)
    std2 = residuals2.std(ddof = 1)

    chi2_object1 = Chi2Regression(linfit1, x_vals1, y_vals1, std1 * np.ones_like(y_vals1))
    chi2_object2 = Chi2Regression(linfit2, x_vals2, y_vals2, std2 * np.ones_like(y_vals1))

    fit1 = iminuit.Minuit(chi2_object1, a = 10, b = 100)
    fit1.errordef = 1
    fit2 = iminuit.Minuit(chi2_object2, a = 10, b = 100)
    fit2.errordef = 1
    print(fit1.migrad())
    print(fit2.migrad())

    Ndof1 = len(y_vals1) - len(fit1.values[:])
    chi21 = fit1.fval
    prop1 = stats.chi2.sf(chi21, Ndof1)
       
    Ndof2 = len(y_vals2) - len(fit2.values[:])
    chi22 = fit2.fval
    prop2 = stats.chi2.sf(chi22, Ndof2)
       
 
    d = {"a1": [fit1.values['a'],fit1.errors['a']], "b1": [fit1.values['b'],fit1.errors['b']], \
               "(fit 1) Ndof":  Ndof1, "(fit 1) chi2": chi21, "(fit 1) Prop": prop1, \
       " a2": [fit2.values['a'],fit2.errors['a']], " b2": [fit2.values['b'],fit2.errors['b']], \
            "(fit 2) Ndof":  Ndof2, "(fit 2) chi2": chi22, "(fit 2) Prop": prop2  } 

    text = nice_string_output(d, extra_spacing=2, decimals=2)
    add_text_to_ax(0.15, 0.95, text, ax0, fontsize=11)
    ax0.set_xlabel(xlabel="Wavelength (nm)", fontsize = 18)
    ax0.set_ylabel(ylabel="Voltage (V)", fontsize = 18)
    ax0.set_xlim((xrange[0], xrange[1]))
    ax0.set_title(label = '2d-histogram of wavelength and voltage distributions', fontsize = 20)


    plt.plot(x_vals1, linfit1(x_vals1, fit1.values['a'], fit1.values['b']), 'k--', linewidth = 0.5, label = 'Fit 1: y = a1 * (x - lambda1) + b1')
    plt.plot(x_vals2, linfit2(x_vals2, fit2.values['a'], fit2.values['b']), 'k--', linewidth = 0.5, label = 'Fit 2: y = a2 * (x - lambda2) + b2')
    plt.legend(fontsize = 15)
    plt.colorbar(im)
    plt.show()




    ## plot data and fit the peaks at ~1280 and ~1875 with Gaussians. 
    N_points = len(wavelength)
    bins = 500
    xmin, xmax = 1200, 2200
    range = [xmin,xmax]
    bin_width = (range[1] - range[0]) /  bins
    plt.hist(lambda_trans2, bins=bins, range=(xmin, xmax), color = 'red', histtype='step', label='histogram', linewidth = 2)
    plt.xlabel(xlabel="Wavelength (nm)", fontsize = 18)
    plt.ylabel(ylabel="Count", fontsize = 18)
    plt.xlim((xmin, xmax))
    plt.title(label = '(Calibrated) Spectral lines of hydrogen', fontsize = 20)
   
    fig, ax = plt.subplots(figsize = (12,6))
   
    counts, edges, _ = ax.hist(wavelength, bins=bins, range=(xmin, xmax), color = 'red', histtype='step', label='histogram', linewidth = 2)
    ax.set_xlabel(xlabel="Wavelength (nm)", fontsize = 18)
    ax.set_ylabel(ylabel="Count", fontsize = 18)
    ax.set_xlim((xmin, xmax))
    ax.set_title(label = 'Spectral lines of hydrogen', fontsize = 20)
    plt.show()
    x_vals = 0.5 * (edges[:-1] + edges[1:])
    y_vals = counts
    y_err = np.sqrt(counts)
    mask = (y_vals > 0)
    N_non_empty = len(mask)

 
       #ax.errorbar(x_vals, counts, y_err, fmt = 'k.', elinewidth=1.5, capsize=1.5, capthick=1)
    shift = 5
    peak_range1 = [1276.2 - shift, 1276.2 + shift]
    peak_range2 = [1865.6 - shift, 1865.6 + shift]
  
    peak_bins1 = np.arange(int((peak_range1[0] - 1200) / bin_width), int((peak_range1[1] - 1200) / bin_width)).astype('int')
    peak_bins2 = np.arange(int((peak_range2[0] - 1200) / bin_width), int((peak_range2[1] - 1200) / bin_width)).astype('int')
     



    def background(x, alpha, beta, a, b):
        return a * (x-xmin) ** 2 + b * (x-xmin) + alpha * np.exp(- np.abs(beta) * (x - xmin))   

    def gaussian_binned(x, A, N, mean, std):
        bin_width = (range[1] - range[0]) /  bins
        return np.abs(A) + N * bin_width * 1 / (np.sqrt(2 * np.pi) * std) * np.exp(-0.5 * (x-mean) ** 2 / std ** 2)


    def double_gaussian_binned(x, alpha, beta, N1, N2, mean1, mean2, std1, std2):
        if ( (x < peak_range1[0]) |  (x > peak_range2[1]) | (peak_range1[1] < x < peak_range2[0]) ):
            return alpha * np.exp(- np.abs(beta) * (x-1200))
        else: 
            bin_width = (range[1] - range[0]) /  bins
            val1 =  N1 * bin_width * 1 / (np.sqrt(2 * np.pi * std1 ** 2 )) * np.exp(-0.5 * (x-mean1) ** 2 / std1 ** 2)
            val2 =  np.abs(N2) * bin_width * 1 / (np.sqrt(2 * np.pi * std2 ** 2)) * np.exp(-0.5 * (x-mean2) ** 2 / std2 ** 2)
            return np.abs(val1 + val2) + alpha * np.exp(-np.abs(beta) * (x-1200))

    param_guesses = [np.array([0.01, -0.1, 575, -0.001,])]
    param_list = []
    stat_list = []
    d = {}

    for i, bin in enumerate(param_guesses):
        chi2_object = Chi2Regression(background, x_vals[mask], y_vals[mask], y_err[mask])
        chi2_object.errordef = 1
        fit = iminuit.Minuit(chi2_object, *param_guesses[i])
        print(fit.migrad())
        if not fit.fmin.is_valid:
            print("Chi square fit didn't converge!")
            print(fit.values[:])
        param_list.append( fit.values[:])
        background_params = fit.values[:].copy()

        x_range = np.linspace(xmin, xmax, 5000)
        #ax.plot(x_range, background(x_range, *background_params[:]), 'k--', linewidth = 2, label = 'Background fit')
        #ax.legend(loc = 'best', fontsize = 16)
    # Get statistics
        Ndof = len(y_vals[mask]) - len(fit.values[:])
        chi2 = fit.fval
        prop = stats.chi2.sf(chi2, Ndof)

        if 0:
            d = {"Entries": len(wavelength), "fitted a": [fit.values['a'],fit.errors['a']], "fitted b": [fit.values['b'],fit.errors['b']], \
                    "fitted alpha": [fit.values['alpha'],fit.errors['alpha']],  "fitted beta": [fit.values['beta'],fit.errors['beta']],\
                            "Ndof": Ndof, "Chi squared": chi2, "Prop": prop} 

            text = nice_string_output(d, extra_spacing=2, decimals=5)
            add_text_to_ax(0.15, 0.95, text, ax, fontsize=14)

   # plt.show()
   # plt.figure()


    def double_gaussian_binned(x, N1, N2, mean1, mean2, std1, std2):   
        bin_width = (range[1] - range[0]) /  bins

        val = background(x, *background_params[:])

        val += N1 * bin_width * 1 / (np.sqrt(2 * np.pi) * std1) * np.exp(-0.5 * (x-mean1) ** 2 / std1 ** 2)
        val += N2 * bin_width * 1 / (np.sqrt(2 * np.pi) * std2) * np.exp(-0.5 * (x-mean2) ** 2 / std2 ** 2)
        return val

  
    param_guesses = [np.array([4000,4000,1275,1865, 3, 3], dtype = 'float')]
    param_list = []
    stat_list = []

    for i, bin in enumerate(param_guesses):
        chi2_object = Chi2Regression(double_gaussian_binned, x_vals[mask], y_vals[mask], y_err[mask])
        chi2_object.errordef = 1
        fit = iminuit.Minuit(chi2_object, *param_guesses[i])
        print(fit.migrad())
        if not fit.fmin.is_valid:
            print("Chi square fit didn't converge!")
            print(fit.values[:])
        param_list.append( fit.values[:])

    # plot fit
        x_range = np.linspace(xmin, xmax, 5000)
    
        fit_vals = double_gaussian_binned(x_range, *fit.values[:])
        ax.plot(x_range, fit_vals,  linewidth = 2, label = f'Fit')

    
    # Get statistics
        Ndof = len(y_vals[mask]) - len(fit.values[:])
        chi2 = fit.fval
        prop = stats.chi2.sf(chi2, Ndof)
       
        # find weighted average of stds to use later
        var = 1/ (1 / fit.errors['std1'] ** 2 + 1 / fit.errors['std2'] ** 2)
        std_fixed = (fit.values['std1']/fit.errors['std1'] ** 2 + fit.values['std2']/fit.errors['std2'] ** 2) * var
        print(std_fixed)

        if 0:
            d = {"Entries": len(wavelength), "fitted N1": [fit.values['N1'],fit.errors['N1']], "fitted N2": [fit.values['N2'],fit.errors['N2']], \
                    "fitted mean1": [fit.values['mean1'],fit.errors['mean1']],  "fitted mean2": [fit.values['mean2'],fit.errors['mean2']],\
                "fitted std1": [fit.values['std1'],fit.errors['std1']], "fitted std2": [fit.values['std2'],fit.errors['std2']], \
                "Ndof": Ndof, "Chi squared": chi2, "Prop": prop} 
    
            text = nice_string_output(d, extra_spacing=2, decimals=2)
            add_text_to_ax(0.15, 0.95, text, ax, fontsize=14)

        
    ax.legend(loc = 'best', fontsize = 16)
    ax.plot(x_range, background(x_range, *background_params[:]), 'k--', linewidth = 2, label = 'Background fit')
    ax.legend(loc = 'best', fontsize = 16)
    fig.tight_layout()
    #plt.show()
   
    
    def many_gaussians_binned(x, params):   
        """
        param is a vector. First entry is N of first gaussian, mu second entry etc. std fixed
        """

        fits = int(len(params) / 2)
        Nparameters = len(params)
      
        bin_width = (range[1] - range[0]) /  bins
        val = background(x, *background_params[:])

        mask_even = np.arange(0, Nparameters, 2)
        mask_odd = np.arange(1, Nparameters, 2)
        N = params[mask_even] 
    
        mean = params[mask_odd]
      
   
        val_vec= N * bin_width * 1 / (np.sqrt(2 * np.pi) * std_fixed) * np.exp(-0.5 * (x - mean) ** 2 / std_fixed ** 2)
        val += val_vec.sum()

      #  val += params[0] * bin_width * 1 / (np.sqrt(2 * np.pi) * std_fixed) * np.exp(-0.5 * (x-params[1]) ** 2 / std_fixed ** 2)
       # val += params[2] * bin_width * 1 / (np.sqrt(2 * np.pi) * std_fixed) * np.exp(-0.5 * (x-params[3]) ** 2 / std_fixed ** 2)
       # for i in range(fits):
        #    val += params[i] * bin_width * 1 / (np.sqrt(2 * np.pi) * std_fixed) * np.exp(-0.5 * (x-params[i+1]) ** 2 / std_fixed ** 2)

        return val


    def many_gaussians_binned_plot(x, params):   
        """
        param is a vector. First entry is N of first gaussian, mu second entry etc. std fixed
        """

        fits = int(len(params) / 2)
        Nparameters = len(params)
      
        bin_width = (range[1] - range[0]) /  bins
        val = background(x, *background_params[:])

      #  mask_even = np.arange(0, Nparameters, 2)
       # mask_odd = np.arange(1, Nparameters, 2)
        #N = params[mask_even] 
        #mean = params[mask_odd]
   
       # val_vec= N * bin_width * 1 / (np.sqrt(2 * np.pi) * std_fixed) * np.exp(-0.5 * (x - mean) ** 2 / std_fixed ** 2)
        #val += val_vec.sum()

      #  val += params[0] * bin_width * 1 / (np.sqrt(2 * np.pi) * std_fixed) * np.exp(-0.5 * (x-params[1]) ** 2 / std_fixed ** 2)
       # val += params[2] * bin_width * 1 / (np.sqrt(2 * np.pi) * std_fixed) * np.exp(-0.5 * (x-params[3]) ** 2 / std_fixed ** 2)
        for i in np.arange(fits):
            val += params[2 * i] * bin_width * 1 / (np.sqrt(2 * np.pi) * std_fixed) * np.exp(-0.5 * (x-params[2 * i+1]) ** 2 / std_fixed ** 2)

        return val







    fig2, ax2 = plt.subplots(figsize = (12,6))
    counts, edges, _ = ax2.hist(wavelength, bins=bins, range=(xmin, xmax), color = 'red', histtype='step', label='histogram', linewidth = 2)
    ax2.set_xlabel(xlabel="Wavelength (nm)", fontsize = 18)
    ax2.set_ylabel(ylabel="Count", fontsize = 18)
    ax2.set_xlim((xmin, xmax))
    ax2.set_title(label = 'Spectral lines of hydrogen', fontsize = 20)

    x_vals = 0.5 * (edges[:-1] + edges[1:])
    y_vals = counts
    y_err = np.sqrt(counts)
    mask = (y_vals > 0)
    N_non_empty = len(mask)



    param_guesses = np.array([4000,1275,4000,1865, 600, 1934, 1000, 2150, 400, 1807, 400, 1726, 300, 1673, 300, 1633, 300, 1602, 300, 1580, 250, 1560, 250, 1550])
 
    param_list = []
    stat_list = []

  
    chi2_object = Chi2Regression(many_gaussians_binned, x_vals[mask], y_vals[mask], y_err[mask])
    chi2_object.errordef = 1
    fit = iminuit.Minuit(chi2_object, param_guesses[:])
    print(fit.migrad())
    if not fit.fmin.is_valid:
        print("Chi square fit didn't converge!")
        print(fit.values[:])
  #  param_list.append( fit.values[:])

    # plot fit
    x_range = np.linspace(xmin, xmax, 5000)
    many_gauss_vec = np.vectorize(lambda x: many_gaussians_binned_plot(x, fit.values[:]))
    fit_vals = many_gauss_vec(x_range)
    ax2.plot(x_range, fit_vals, 'g-',  linewidth = 2, label = f'Fit')

    
    # Get statistics
    Ndof = len(y_vals[mask]) - len(fit.values[:])
    chi2 = fit.fval
    prop = stats.chi2.sf(chi2, Ndof)
       # d = {}
        #fits = int(len(fit.values[:]) / 2 )
    if 0:
        d = {"Entries": len(wavelength), "fitted N1": [fit.values['N1'],fit.errors['N1']], "fitted N2": [fit.values['N2'],fit.errors['N2']], \
                "fitted mean1": [fit.values['mean1'],fit.errors['mean1']],  "fitted mean2": [fit.values['mean2'],fit.errors['mean2']],\
            "fitted std1": [fit.values['std1'],fit.errors['std1']], "fitted std2": [fit.values['std2'],fit.errors['std2']], \
            "Ndof": Ndof, "Chi squared": chi2, "Prop": prop} 

        text = nice_string_output(d, extra_spacing=2, decimals=2)
        add_text_to_ax(0.15, 0.95, text, ax, fontsize=14)

        
    ax2.legend(loc = 'best', fontsize = 16)
    ax2.plot(x_range, background(x_range, *background_params[:]), 'k--', linewidth = 2, label = 'Background fit')
    ax2.legend(loc = 'best', fontsize = 16)
    fig2.tight_layout()

    plt.show()

    params = np.array(fit.values[:])
    err_params = np.array(fit.errors[:])
    N_vals = params[np.arange(0,len(params),2)]
    N_err = err_params[np.arange(0,len(params),2)]
    lambda_vals =  params[np.arange(1,len(params),2)]
    lambda_err =  err_params[np.arange(1,len(params),2)]
  
      ## define calibration function
    x_fit = np.array([1276.16, 1865.66])
    x_bohr = np.array([1282.174, 1875.637])
    dx_fit = np.array([0.07, 0.04])
    dy_bohr = np.array([0.005, 0.005])
    a = (x_bohr[1] - x_bohr[0]) / (x_fit[1] - x_fit[0])
    b = x_bohr[1] - a * x_fit[1]
    da = 0
    db = 0.04

    calibration = lambda x: a * x + b
    dcal = lambda dx: np.sqrt(a**2 * dx ** 2 + db ** 2)


    for i in np.arange(12):
        print(i)
        print(lambda_vals[i], "\u00B1", lambda_err[i])
        print("lambda cal", calibration(lambda_vals[i]), "\u00B1", dcal(lambda_err[i]))
       # print(N_vals[i], "\u00B1", N_err[i], "  ", N_vals[i] / N_err[i], " ", stats.norm.sf(N_vals[i] / N_err[i]))


  







    
    if 0:
        peak_ranges = [peak_range1, peak_range2]
        peak_bins = [peak_bins1, peak_bins2]
        param_guesses = [np.array([250,4000,1275,3], dtype = 'float'), \
            np.array([150, 4000, 1865, 3], dtype = 'float')]
        param_list = []
        stat_list = []
        d = {}

        for i, bin in enumerate(peak_bins):
            chi2_object = Chi2Regression(gaussian_binned, x_vals[bin], y_vals[bin], y_err[bin])
            chi2_object.errordef = 1
            fit = iminuit.Minuit(chi2_object, *param_guesses[i])
            fit.migrad()
            if not fit.fmin.is_valid:
                print("Chi square fit didn't converge!")
                print(fit.values[:])
            param_list.append( fit.values[:])

        # plot fit
            x_range = np.linspace(peak_ranges[i][0], peak_ranges[i][1], 1000)
        
            fit_vals = gaussian_binned(x_range, *fit.values[:])
            ax.plot(x_range, fit_vals, linewidth = 2, label = f'Fit {i+1}')

        # Get statistics
            Ndof = len(y_vals[bin]) - len(fit.values[:])
            chi2 = fit.fval
            prop = stats.chi2.sf(chi2, Ndof)
            stat_list.append([Ndof, chi2, prop])
            

            d_add = {f"FIT {i+1}": "  ", f"A{i} (fit)": [fit.values['A'],fit.errors['A']] ,f"N{i+1} (fit)": [fit.values['N'],fit.errors['N']], \
                        f"Mean{i+1} (fit)": [fit.values['mean'],fit.errors['mean']], \
                    f"Std{i+1} (fit)": [fit.values['std'],fit.errors['std']], \
                    f"Ndof{i+1}": Ndof, f"Chi squared {i+1}": chi2, f"Prop{i+1}": prop, " ": " "} 
        
            d.update(d_add)

        text = nice_string_output(d, extra_spacing=2, decimals=2)
        add_text_to_ax(0.15, 0.95, text, ax, fontsize=10)

        ax.legend(loc = 'best')

        fig.tight_layout()
        plt.show()

# Set plotting style
sns.set_theme()
sns.set_style("darkgrid")
sns.set_context("paper") #Possible are paper, notebook, talk and poster
rcParams['lines.linewidth'] = 2 
rcParams['axes.titlesize'] =  18
rcParams['axes.labelsize'] =  18
rcParams['xtick.labelsize'] = 10
rcParams['ytick.labelsize'] = 10
rcParams['legend.fontsize'] = 15
rcParams['font.family'] = 'serif'
rcParams['figure.figsize'] = (6,6)
rcParams['axes.prop_cycle'] = cycler(color = ['teal', 'navy', 'coral', 'plum', 'purple', 'lightblue', 'olivedrab', 'black'])
np.set_printoptions(precision = 5, suppress=1e-10)

## Set parameters and which problems to run
p2_1, p2_2, p3, p4_1, p4_2, p5_1, p5_2 = False, False, False, False, True, False, False




def main():
    
    problem_numbers = [p2_1, p2_2, p3, p4_1, p4_2, p5_1, p5_2]
    f_list = [P2_1, P2_2, P3, P4_1, P4_2, P5_1, P5_2]
    names = ['p2_1', 'p2_2', 'p3', 'p4_1', 'p4_2', 'p5_1', 'p5_2']

    for i, f in enumerate(f_list):
        if problem_numbers[i]:
            print(f'\nPROBLEM {names[i][1:]}:')
            f()
   


    if 0:
        ## Load data [time in s and area in cm^2]
        time, area, area_err = np.loadtxt('data_AlgaeGrowth.csv', skiprows = 1, unpack = True, delimiter = ',')
        assert (time.shape == area.shape == area_err.shape)

        N = 6

        x = np.ones(2*N)
        x[np.arange(0,9,2)] = -1

        runs = len(x)
        print("exp ", runs)

        z,p = runstest_homebrew(x)

        print (x)



if __name__ == '__main__':
    main()

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
        N_non_empty = len(counts[mask])

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

def anderson_ks_fisher():

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


    counts, edges, _ = ax.hist(residuals, bins = bins, range = range,
     histtype = 'stepfilled', lw = 2, alpha = 0.5, label = 'Histogram of time residuals')
    

    x_vals = 0.5 * (edges[:-1] + edges[1:])
    errors = np.sqrt(counts)
    mask = (counts > 0)
    N_non_empty = len(mask)

    ax.set(xlabel = 'Time residual (s)', ylabel = 'Count')
    ax.errorbar(x_vals, counts, errors, fmt = 'k.', elinewidth=1, capsize=1, capthick=1)

   
    parameter_guesses = np.array([1500, 0, 0.2])
    fit_chi2 = do_chi2_fit(gaussian_binned, x_vals[mask], counts[mask], errors[mask], parameter_guesses)
    x_fit = np.linspace(range[0], range[1], 1000)

    if 0:
        fit_LH = do_LH_fit(gaussian_LH, residuals , fit_chi2.values, bound = range, unbinned = True, extended = True)
        fmax = gaussian_LH(fit_LH.values['mean'], *fit_LH.values)
        Nsimulations = 100
        LL_values, p_val = evaluate_likelihood_fit(gaussian_LH, fmax = fmax, parameter_val_arr = fit_LH.values, \
            log_likelihood_val = fit_LH.fval, bounds = range, Ndatapoints = len(residuals), Nsimulations = Nsimulations)

        plot_likelihood_fits(LL_values, p_val, log_likelihood_val = fit_LH.fval, Nsimulations = Nsimulations)
        
        fit_vals_LH = gaussian_binned(x_fit, *fit_LH.values[:])
        ax.plot(x_fit, fit_vals_LH, label = 'LH Gaussian fit')
  
    
    fit_vals_chi2 = gaussian_binned(x_fit, *fit_chi2.values[:])
    ax.plot(x_fit, fit_vals_chi2, label = r'Chi2 Gaussian fit')
    
    d = generate_dictionary(fit_chi2, Ndatapoints = N_non_empty, chi2_suffix = 'chi2')

  
    text = nice_string_output(d, extra_spacing=0, decimals=2)
    add_text_to_ax(0.05, 0.9, text, ax, fontsize=14)

    ax.legend()
    fig.tight_layout()
    plt.show()

def template_one_and_two_plots_fit_points():
        ## Load data x is time in month, y is monthly income in M$, dy is error on income.
    x, y, dy = np.loadtxt('data_LukeLightningLights.txt', usecols = (0,2,3), unpack = True)
    assert (x.shape == y.shape == dy.shape)

    ### STEP 1: Plot the thing
    fig0, ax0 = plt.subplots()
    ax0.errorbar(x, y, dy, fmt = 'k.', elinewidth=1.5, capsize=1.5, capthick=1)
    ax0.set(xlabel = 'Time (months)', ylabel = 'Monthly income (M$)', title = "LLL's monthly income over time")
 
  
    def fit_func(x,a):
        return a

 
    param_guess = np.array([-0.35])
    fit = do_chi2_fit(fit_func, x, y, dy, param_guess)
 
    x_vals = np.linspace(x[0], x[-1], 500)
    ax0.plot(x_vals, fit_func(x_vals, fit_func(x_vals, *fit.values[:]), label = r"")
   


    d = generate_dictionary(fit, Ndatapoints = len(y), chi2_suffix = 'const')
    # Plot figure text
    text = nice_string_output(d, extra_spacing=0, decimals=2)
    add_text_to_ax(0.05, 0.7, text, ax0, fontsize=13)

    ax0.legend()
    fig0.tight_layout()
    plt.show()




def template_two_plots_fit_points():



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
    d = {}

    for i, fit_func in enumerate(func_list):

        ax[i].set(xlabel = "", ylabel = "", title = "")
        ax[i].errorbar(time, area, area_err, fmt = 'r.', elinewidth=1, capsize=1, capthick=1)

        # Optimize parameters according to chi2 value
        fit = do_chi2_fit(fit_func, time, area, area_err, param_guesses[i])
        d0 = generate_dictionary(fit, len(area), chi2_suffix = 'Fit {i}')
        
        # plot fit
        x_range = np.linspace(range[0], range[1], 1000)
        ax[i].plot(x_range, fit_func(x_range, *fit.values[:]), 'k-', linewidth = 2, label = f'{fit_list[i]}')
        ax[i].legend()

    
        text = nice_string_output(d, extra_spacing=2, decimals=2)
        add_text_to_ax(0.35, 0.35, text, ax[i], fontsize=13)

    fig.tight_layout()
    plt.show()

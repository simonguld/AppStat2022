## Author: Simon Guldager Andersen
## Date (latest update): 12/12-2022

### SETUP -----------------------------------------------------------------------------------------------------------------------------------

## Imports:
import numpy as np
import os, sys, itertools
import iminuit as Minuit
from scipy import stats
import matplotlib.pyplot as plt


sys.path.append('C:\\Users\\Simon\\PycharmProjects\\Projects\\Projects\\AppStat2022\\External_Functions')
from ExternalFunctions import Chi2Regression, BinnedLH
from ExternalFunctions import nice_string_output, add_text_to_ax # useful functions to print fit results on figure


## Paths:
# Change path to current directory
path = os.getcwd()
path = os.path.join(path, "Projects\\AppStat2022\\Project")

### FUNCTIONS -------------------------------------------------------------------------------------------------------------------------------



### MAIN ------------------------------------------------------------------------------------------------------------------------------------



count = [None] * 4
times = [None] * 4
names = ['simon', 'maxi', 'masoumeh', 'yasmine']
period_list = []
period_err_list = []
for i, name in enumerate(names):
    row_skip = 0
    if i == 0:
        row_skip = 1
    count[i], times[i] = np.loadtxt(f'{path}\\timer_output_{name}.dat', unpack=True, skiprows = row_skip)
    if name == 'yasmine':
        #account for missing a count
        count[i][18:] +=1
    if i != 0:
        count[i] = np.delete(count[i], - 1)
        times[i] = np.delete(times[i], -1)



def main():
    # Plot data
    plt.style.use('seaborn')
    fig, ax = plt.subplots(nrows = 2, ncols = 2, figsize = (8,8))
    ax = ax.flatten()

    for i in range(4):
        ax[i].plot(count[i], times[i], 'ro', markersize = 5)
        ax[i].set(xlabel = 'Period number', ylabel = 'Total time (s)', title = 'Total elapsed time as a function of period number', ylim=(-25,np.max(times[i])+25))

           # Fit data with a line, at first ignoring the unkown uncertainties

        def line_fit (x, a, b):
            return np.sum(np.abs((x - a * count[i] - b)))

        def line(x, a, b):
            return a * x + b

    # line_vec = np.vectorize(line_fit)

        fitLin = Minuit.Minuit(lambda a,b: line_fit(times[i], a, b), a = 10, b = 0)
        fitLin.errordef = 1

        fitLin.migrad()
        print(fitLin.values['a'], fitLin.values['b'])
        fit_vals = count[i] * fitLin.values['a'] + fitLin.values['b']
        residuals = times[i] - fit_vals

        # Create new y-axis
        ax2 = ax[i].twinx()
        shift = 0
        ax2.set(ylabel = 'Residual time (s)', ylim = (-shift-0.2,-shift+3))


        ax2.set_yticks(np.round(np.arange(-0.3,0.3,0.1)-shift,3), np.round(np.arange(-0.3,0.3,0.1),3), size = 10)

        std_residuals = residuals.std(ddof = 1)
      
        ax2.plot([count[i][0], count[i][-1]], [-shift,-shift], 'k-', linewidth = 2)
        ax2.plot([count[i][0], count[i][-1]], [std_residuals-shift, std_residuals-shift], 'k--', linewidth = 1)
        ax2.plot([count[i][0], count[i][-1]], [-std_residuals-shift, -std_residuals-shift], 'k--', linewidth = 1)
        ax2.errorbar(count[i], residuals-shift, std_residuals, fmt='ro', ecolor='k', elinewidth=1, capsize=2, capthick=1, markersize = 4, label = 'Time Residuals')

        # Fit points again with uncertainties
        chi2_object = Chi2Regression(line, count[i], times[i], std_residuals * np.ones_like(times[i]))
        print(residuals, "\n \n")
        fitLin2 = Minuit.Minuit(chi2_object, a = fitLin.values['a'], b =  fitLin.values['b'])
        fitLin2.errordef = 1
        fitLin2.migrad()
        fit_vals2 = count[i] * fitLin2.values['a'] + fitLin2.values['b']
        period_list.append(fitLin2.values['a'])
        period_err_list.append(fitLin2.errors['a'])

        chi2 = fitLin2.fval
        Ndof = len(count[i]) - 2
        prop = stats.chi2.sf(chi2, Ndof)

        ax[i].plot(count[i], fit_vals2, 'k-', linewidth = 2)
        ax2.legend(loc = 'center left')
        d = {"Offset = ": [fitLin2.values['b'], fitLin2.errors['b']],  "Period = ": [fitLin2.values['a'], fitLin2.errors['a']], "Data points": len(count[i]), "Chi2": chi2, 
        "Prop": prop}

        print(f'{i}    ', residuals.mean(), "  ", std_residuals)
        text = nice_string_output(d, extra_spacing=2, decimals=4)
        add_text_to_ax(0.02, 0.97, text, ax[i], fontsize=13)


    ## Print out periods
    for i in range(len(period_list)):
        print("Period: ", period_list[i], "\u00B1", period_err_list[i])

    ## Including measurement 4:
    av_total = np.average(period_list)
    
    chi2_tot = np.sum(np.power((period_list-av_total) / period_err_list, 2)  )
    prop_tot = stats.chi2.sf(chi2_tot, Ndof)
    ## Dropping measurement 4:
    av_partial = np.average(period_list[0:3])
    chi2_partial = np.sum(np.power((period_list[0:3]-av_total) / period_err_list[0:3], 2)  )
    prop_partial = stats.chi2.sf(chi2_partial, Ndof)

    print("Average of all measurements, with std calculated from the discrepancy: ", f'{np.average(period_list):6.6f}, ', \
            f'std:  {np.std(period_list, ddof = 1):6.6f}' ) 
    print("Average of 3 best measurements, with std calculated from the discrepancy: ", f'{np.average(period_list[0:3]):6.6f}, ', \
            f'std:  {np.std(period_list[0:3], ddof = 1):6.6f}' )
    print("Chi squared and prop all measurements: " f'{chi2_tot:6.1e}, ', prop_tot)
    print("Chi squared and prop exluding last measurement: ", f'{chi2_partial:6.1e}', prop_partial)


    #plt.legend(loc = 'center left')
    fig.tight_layout()
    plt.show()
 
    

    

if __name__ == '__main__':
    main()

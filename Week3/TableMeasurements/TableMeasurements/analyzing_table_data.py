### SETUP -----------------------------------------------------------------------------------------------------------------------------------

## Imports:
import os, sys, itertools
import numpy as np
import iminuit as Minuit
import matplotlib.pyplot as plt
from scipy import stats

sys.path.append('C:\\Users\\Simon\\PycharmProjects\\Projects\\Projects\\AppStat2022\\External_Functions')
from ExternalFunctions import Chi2Regression, BinnedLH
from ExternalFunctions import nice_string_output, add_text_to_ax # useful functions to print fit results on figure

### FUNCTIONS -------------------------------------------------------------------------------------------------------------------------------




### MAIN ------------------------------------------------------------------------------------------------------------------------------------

## Loading data
path = os.getcwd()
path = os.path.join(path, "Projects\\AppStat2022\\Week3\\TableMeasurements\\TableMeasurements\\Data")

years = range(2009, 2023)

data_30 = [None] * len(years)
err_30 = [None] * len(years)
data_2 = [None] * len(years)
err_2 = [None] * len(years)

years = np.arange(2009, 2023)

for i, year in enumerate(years):
    data_30[i], err_30[i], data_2[i], err_2[i] = np.loadtxt(f'{path}\\data_TableMeasurements{year}.txt', unpack = True, skiprows = 2)



def main():
    ## Step 1: Plot all measurements for each year
    year = 2015
    year_index = int(np.argwhere(years == year).flatten())

  
    hist30 = False

    if hist30:
        fig, ax = plt.subplots(nrows = 4, ncols = 4, figsize = (6,12))
        ax = ax.flatten()

        for i, year in enumerate(years):

            bins = int(len(data_30[i]) / 2)
            range = (3,3.8)

            ax[i].hist(data_30[i], bins = bins, range = range, histtype='step', linewidth = 2)

            ax[i].set(xlabel = 'length in meters', ylabel = 'No. of measurements', title = f'Histogram (30cm) for {year}')

            d = {'Entries': len(data_30[i]), 'Mean': np.mean(data_30[i]), 'Std': np.std(data_30[i], ddof = 1)}
            text = nice_string_output(d, extra_spacing=2, decimals=3)
            add_text_to_ax(0.02, 0.97, text, ax[i], fontsize=10);

        fig.tight_layout()
        plt.show()

    # Plot all measurements for all years
    fig2, ax2 = plt.subplots(nrows = 2, ncols = 2, figsize = (10,10))
    ax2 = ax2.flatten()

    # Merge values to a single list
    if 1:
        data30 = np.array(list(itertools.chain(*data_30)))
        data2 = np.array(list(itertools.chain(*data_2)))
        err30 = np.array(list(itertools.chain(*err_30)))
        err2 = np.array(list(itertools.chain(*err_2)))

    # Remove data points with negative uncertainties
    neg_index30 = (err30 > 0)
    neg_index2 = (err2 > 0)
    print(f'No. of points for 30 cm with negative uncertainties: ', neg_index30.sum())
    print(f'No. of points for 30 cm with negative uncertainties: ', neg_index2.sum())
    print("These uncertainties will be set to 0")

    err30 = err30[neg_index30]
    err2 = err2[neg_index2]
    data30 = data30[neg_index30]
    data2 = data2[neg_index2]

    # if err > 0.2 = 20 cm, we assume that the observer has reported the uncertainty in centimeters instead of meters.
    err30[(err30 > 0.2)] = err30[(err30 > 0.2)] / 100
    err2[(err2 > 0.2)] = err2[(err2 > 0.2)] / 100

    # Discard all measurements less than 2
    big_enough30 = (data30 > 2)
    big_enough2 = (data2 > 2)
    data30 = data30 [big_enough30]
    err30 = err30 [big_enough30]
    data2 = data2[big_enough2]
    err2 = err2[big_enough2]

    # discard all measurements more than 3.5 std away from mean
    p = 2 * stats.norm.sf(3.5)
    print(f"Probability that any of the {data30.size} points for the 30 data will be further than 4 standarddeviations = {3.5*data30.std(ddof = 1)} \
    away from the mean = {data30.mean()}: ", np.round(data30.size * p,6))
    print(f"Probability that any of the {data2.size} points for the 30 data will be further than 4 standarddeviations = {3.5*data2.std(ddof = 1)} \
    away from the mean = {data2.mean()}: ", np.round(data2.size * p,6))
    print('...such points will be discarded')

    # discard all measurements more than 4 std (their own) away from mean, i.e. overly precise outlier points
    mask30 = (np.abs((data30 - data30.mean())) / err30 < 4)
    mask2 = (np.abs((data2 - data2.mean())) / err2 < 4)
    data30 = data30[mask30]
    err30 = err30[mask30]
    data2 = data2[mask2]
    err2 = err2[mask2]

    # Modify uncertainties: There is no way a 30 cm ruler measurement can be done more precisely than to 1 cm uncertaintiy. Similarly with 1 mm for 200 cm ruler. 
    err30[(err30 < 0.005)] = 0.005
    err2[(err2 < 0.001)] = 0.001

    mean30 = data30.mean()
    std30 = data30.std(ddof = 1)
    mean2 = data2.mean()
    std2 = data2.std(ddof = 1)

    mask30 = (data30 > mean30 - 3.5 * std30) & (data30 < mean30 + 3.5 * std30)
    mask2 = (data2 > mean2 - 3.5 * std2) & (data2 < mean2 + 3.5 * std2)

    data30 = data30[mask30]
    err30 = err30[mask30]
    data2 = data2[mask2]
    err2 = err2[mask2]


    # There are two bumbs 30 cm to the left and right of the mean for the 30 cm ruler data, consistent with observers miscounting by 1. These data are translated 
    # by 30 cm to account for this systematic error
    mean30 = data30.mean()
    mask_bottom = (data30 > mean30 - 0.35) & (data30 < mean30 - 0.25)
    mask_top = (data30 > mean30 + 0.25) & (data30 < mean30 + 0.35)
    data30[mask_bottom] = data30[mask_bottom] + 0.30
    data30[mask_top] =   data30[mask_top] - 0.30


    data = [data30, data2, err30, err2]
    lengths = [30, 200]

    # when making histogram, consider only the following ranges
    range = ((3.2,3.5), (3.25,3.45))
    bins = (60, 40)

    Minuit.print_level = 1
    for i, length in enumerate(lengths):


        ## Plot histogram
        counts, bin_edges, _ = ax2[2*i].hist(data[i], bins = bins[i], range = range[i], histtype = 'step', linewidth = 2)
        ax2[2*i].set(xlabel = 'length in meters', ylabel = 'No. of measurements', title = f'Histogram({length} cm) for all years')

      
        ## Fit histogram
        s = np.sqrt(counts)
        x_values = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        y_mean = np.average(data[i])
        y_val = counts[counts >0]
        bin_width = (range[i][1] - range[i][0]) / bins[i]
    

        def gaussian_fit(x, N, mean, std):
            return N * bin_width * 1/(std * np.sqrt(2 * np.pi)) * np.exp(-0.5 * (x-mean)**2 / std**2)

      

        chi2_object = Chi2Regression(gaussian_fit, x_values[counts >0], y_val, s[counts>0])
        chi2_object.errordef = 1
    
    
        minuit_hist = Minuit.Minuit(chi2_object, N = y_val.sum(), mean = y_mean, std = y_val.std(ddof = 1))
        #minuit_hist.errordef = 1
       # minuit_hist.migrad()
        print(minuit_hist.migrad())

        # Plot fit
        x_values = np.linspace(range[i][0],range[i][1],1000)

        ax2[2*i].plot(x_values, gaussian_fit(x_values, *minuit_hist.values[:]), 'k-', linewidth = 3)

        Ndof = y_val.size - 3
        chi2 = minuit_hist.fval
        propability = stats.chi2.sf(chi2,Ndof)

        d = {'Entries': y_val.sum(), 'Mean': np.mean(data[i]), 'Std': np.std(data[i], ddof = 1), 'Fit entries': [minuit_hist.values['N'], minuit_hist.errors['N']],
            'Fit mean': [minuit_hist.values['mean'], minuit_hist.errors['mean']], 'Fit std': [minuit_hist.values['std'], minuit_hist.errors['std']],
            'Chi2': chi2, 'Propability': propability}

        text = nice_string_output(d, extra_spacing=2, decimals=3)
        add_text_to_ax(0.02, 0.97, text, ax2[2*i], fontsize=8)


        ## Plot scatterplot
        ax2[2*i+1].errorbar(np.arange(data[i].size), data[i], data[i+2], fmt='r.', ecolor='k', elinewidth=0.4, capsize=0.05, capthick=0.05, markersize = 1)

        ax2[2*i+1].set(xlabel = 'Measurement number', ylabel = 'Length', title = f'Scatterplot ({length} cm) for all years')

        ## Fit scatterplot
        def const_fit(x,mean):
            return mean * x ** 0

        chi2_object = Chi2Regression(const_fit, np.arange(data[i].size), data[i], data[i+2])
        chi2_object.errordef = 1
        minuit_scatter = Minuit.Minuit(chi2_object, mean = y_mean)
        minuit_scatter.migrad()
        chi2 = minuit_scatter.fval
        Ndof = data[i].size - 1
        propability = stats.chi2.sf(chi2, Ndof)

        # Plot fit
        data_no = np.arange(data[i].size)

        ax2[2*i + 1].plot(data_no, const_fit(data_no, minuit_scatter.values[:]), 'k-', linewidth = 2)

        d = {'Entries': data[i].size, 'Mean': np.mean(data[i]), 'Std': np.std(data[i], ddof = 1), 'Fit mean': [minuit_scatter.values['mean'], minuit_scatter.errors['mean']],
            'Chi2': chi2, 'Propability': propability}
        text = nice_string_output(d, extra_spacing=2, decimals=3)
        add_text_to_ax(0.02, 0.97, text, ax2[2*i+1], fontsize=8)

    fig2.tight_layout()
    plt.show()





if __name__ == '__main__':
    main()

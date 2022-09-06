#!/usr/bin/env python

# © Saket Navlakha, Cold Spring Harbor Laboratory
# Released: September 6, 2022.
# Please cite: Dasgupta, Hattori, Navlakha. Nature Communications, 2022. 

import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
from scipy.optimize import curve_fit
from sklearn import metrics
from matplotlib import rcParams
import scipy.stats
import numpy as np
import pandas as pd

np.random.seed(55)

# Set plotting style.
plt.style.use('ggplot')
plt.rc('font', **{'sans-serif':'Arial', 'family':'sans-serif'})
rcParams.update({'font.size': 28})
plt.rcParams["figure.figsize"] = (13,10) # (16,10) for fig7 to make room for the colorbar!


# Constants.
NUM_CELLS   = 72
NUM_TRIALS  = 10
BOOTSTRAP   = 20000
DIM_ALPHA   = 0.16
MARKER_SIZE = 10
PFONT       = 19

# Fit for the exponential function.
def exp_fit(x, a, b):
    y = a*np.exp(b*x)
    return y


def fig1(D):
    """ Figure 1: number of trials (count category) vs MBON response. 
            Draws raw data and median statistic + bootstrap confidence intervals.
            Plots exponential fit and statistics.
    """

    # Plot the raw data.
    x = range(1,NUM_TRIALS+1)
    for i in range(NUM_CELLS):
        plt.plot(x,D[i,:],linewidth=1,marker='o',markersize=1,alpha=DIM_ALPHA,c='tab:blue',zorder=1)


    # Calculate median per trial.
    Medians = np.median(D,axis=0)


    # Compute confidence intervals for each trial.
    # From: https://towardsdatascience.com/calculating-confidence-interval-with-bootstrapping-872c657c058d
    cis = np.empty((2,NUM_TRIALS)) # (lower,upper) for each trial.
    for trial in range(NUM_TRIALS):
        bs_replicates = np.empty(BOOTSTRAP)

        # For each bootstrap, create a random sample, and compute the median.
        for i in range(BOOTSTRAP):
            bs_sample = np.random.choice(D[:,trial],size=NUM_CELLS,replace=True)
            bs_replicates[i] = np.median(bs_sample)

        # Compute 99% confidence intervals: The first line returns the actual interval.
        # For example, if the median is 0.413, then this returns [0.342, 0.513]. 
        # The second line subtracts the median from lower and upper bounds to get the 
        # actual length of the errorbars. Absolute values are taken because the lengths 
        # should be positive. 
        lower,upper = np.percentile(bs_replicates,[0.5,99.5]) 
        cis[:,trial] = [abs(Medians[trial]-lower),abs(Medians[trial]-upper)]

        print("Trial %i, Median = %.3f" %(trial+1,Medians[trial]))
        
    # Plot the median along with the error bar.
    plt.errorbar(x,Medians,yerr=cis,marker='o',markersize=MARKER_SIZE,linewidth=0,c='tab:blue',elinewidth=3,label="data",zorder=100)


    # Fit the exponential function to the Medians.
    # From: https://blog.finxter.com/exponential-fit-with-scipys-curve_fit/
    fit = curve_fit(exp_fit,x,Medians)
    fit_eq = fit[0][0]*np.exp(fit[0][1]*np.arange(min(x),max(x),0.01))
    print("Exp fit:")
    print("   a = %.3f, b = %.3f" %(fit[0][0],fit[0][1]))

    # Compute the R^2 and suppression constant.
    # From: https://stackoverflow.com/questions/19189362/getting-the-r-squared-value-using-curve-fit
    y_pred = exp_fit(x, fit[0][0],fit[0][1])
    r2 = metrics.r2_score(Medians, y_pred)
    print("   R^2 = %.3f" %(r2))
    print("   Suppression constant = %.2f" %(y_pred[1]/y_pred[0]))

    # Plot exponential fit.
    plt.plot(np.arange(min(x),max(x),0.01),fit_eq,c='tab:red',alpha=0.75,linewidth=4,label=r'exp fit ($R^2=%.3f$)' %(r2),zorder=10)


    # Plot parameters.
    plt.legend()
    plt.xlim(0.90,10.10)
    plt.ylim(-0.10,1.02)
    plt.xticks(x)
    plt.xlabel("# of times odor experienced\n(Count category)",labelpad=20)
    plt.ylabel(r'MBON-$\alpha\'3$ response' + "\n" + '(norm. int. $\Delta$F/F)')
    plt.savefig("../figs/data_suppression.pdf",bbox_inches='tight')
    plt.close()


def fig3(D):
    """ Figure 3: plots kde histogram of responses for each count category. """
    
    # D[D < 0] = 0 # get rid of negative values -- causes artificial spike at x=0

    # Get responses for each count category.
    response = {} # count category i -> list of responses
    for i in range(1,NUM_TRIALS+1): response[i] = D[:,i-1]

    # Make histogram.
    responses = (response[i] for i in range(1,NUM_TRIALS+1))
    bins=np.histogram(np.hstack(responses), bins=40)[1] 

    alpha = 0.2
    colors = ['tab:blue','tab:orange','tab:red','tab:green','k','k','k','k','k','k']

    # Used to make broken y-axis, since category 0 has massively larger responses.
    f, (ax1, ax2) = plt.subplots(ncols=1, nrows=2, sharex=True, gridspec_kw={'height_ratios':[1, 4]})

    # Plot the distribution of each category.
    for i in range(1,NUM_TRIALS+1):
        if i <= 4: label = i
        elif i == 5: label = "5-10"
        else: label = ""

        # Plot the data on both axes.
        ax = sns.histplot(response[i],alpha=alpha,bins=bins,kde=True,stat='probability',color=colors[i-1],label=label,line_kws={"linewidth": 5, "alpha": 1},ax=ax1)
        ax = sns.histplot(response[i],alpha=alpha,bins=bins,kde=True,stat='probability',color=colors[i-1],label=label,line_kws={"linewidth": 5, "alpha": 1},ax=ax2)


    plt.xlabel(r'MBON-$\alpha\'3$ response' + "\n" + '(norm. int. $\Delta$F/F)')
    plt.xlim(-0.45,1.01)

    # For the bottom plot.
    ax2.set_ylabel("Probability")
    ax2.legend(loc=(0.6,0.51))
    ax2.set_ylim(-0.01,0.35)
    
    # For the top plot.
    ax1.set_ylabel("")
    ax1.set_ylim(0.35,1.00)
    
    plt.subplots_adjust(hspace=0.06)
    plt.savefig("../figs/dist_broken.pdf",bbox_inches='tight')
    plt.close()


def fig4(D):
    """ Figure 4: plots heatmap of p-values between all pairs of count categories. """

    pvals = np.ones((10,10)) # 10x10 matrix of p-values.
    for i in range(0,NUM_TRIALS):
        for j in range(i+1,NUM_TRIALS):
            pval = scipy.stats.ranksums(D[:,i],D[:,j])[1]

            pvals[j,i] = pval
            

    # Annotations for each cell with the p-value.
    annot = [[f"{pvals[i,j]:.1e}" if i>=j else "" for j in range(10)] for i in range(10)]
    
    # Used to color the boxes as red (significant) and white (not significant)
    pvals[pvals < 0.01] = 0            
    pvals[pvals >= 0.01] = 1
    cmap = colors.ListedColormap(['tab:red', 'tab:gray'])

    matrix = np.triu(pvals) # mask the upper triangle. 

    # Plot the heatmap.
    g=sns.heatmap(pvals,cmap=cmap,linewidth=2,annot=annot,fmt='',cbar=False,annot_kws={"size": 17},mask=matrix)

    g.set_facecolor('0.96') # set the background color of unused cells to this color.
    
    # change to 1-2-3-many.
    tickslabels=["1","2","3","4","5","6","7","8","9","10"]
    plt.xticks(ticks=np.arange(0.5,10.5,1.0),labels=tickslabels)
    plt.yticks(ticks=np.arange(0.5,10.5,1.0),labels=tickslabels)

    plt.yticks(rotation=0) # make the y-tick labels straight instead of horizontal.
    plt.ylabel("Count category",labelpad=10)
    plt.xlabel("Count category")
    plt.savefig("../figs/heatmap_pvalues.pdf",bbox_inches='tight')
    #plt.show()
    plt.close()


def fig7(D):
    """ Figure 7: Heatmap of ROC AUC of all pairwise tests. """

    aucs = np.ones((NUM_TRIALS,NUM_TRIALS)) # 10x10 matrix of p-values.

    # Compute auROC for every pair of trials.
    for i in range(0,NUM_TRIALS):
        for j in range(i+1,NUM_TRIALS):

            # Get the scores for i and j only.
            y_scores = D[:,[i,j]]

            # The labels will be 1 for i and 0 for j.
            y_true = np.zeros((NUM_CELLS,2))
            y_true[:,0] = 1

            # Flatten both matrices into arrays.
            y_true   = y_true.flatten()
            y_scores = y_scores.flatten()

            assert sum(y_true) == NUM_CELLS # half 1s and half 0s.

            # Compute the ROC curve and AUC.
            fpr, tpr, thresholds = metrics.roc_curve(y_true, y_scores)
            auc = metrics.auc(fpr, tpr)

            # Print output and find the threshold that maximizes the tpr - fpr rate.
            print("Category %i vs %i\tAUC=%.3f\tThr=%.3f" %(i,j,auc,thresholds[np.argmax(tpr - fpr)]))

            # Store auc for the heatmap.
            aucs[j,i] = auc            
    

    # Annotations for each cell with the p-value.
    annot = [[f"{aucs[i,j]:.3f}" if i>=j else "" for j in range(NUM_TRIALS)] for i in range(NUM_TRIALS)]
    
    matrix = np.triu(aucs) # mask the upper triangle. 

    # Used to color the boxes as an interpolation between two colors.
    cmap = colors.LinearSegmentedColormap.from_list('', ['tab:gray','tab:red'], N=4)

    # Plot the heatmap -- for interpolating colors.
    g=sns.heatmap(aucs,linewidth=2,annot=annot,fmt='',cbar=True,annot_kws={"size": 17},mask=matrix,vmin=0.5,vmax=1.0,cmap=cmap)

    # change to 1-2-3-many.
    tickslabels=["1","2","3","4","5","6","7","8","9","10"]
    plt.xticks(ticks=np.arange(0.5,10.5,1.0),labels=tickslabels)
    plt.yticks(ticks=np.arange(0.5,10.5,1.0),labels=tickslabels)

    g.set_facecolor('0.96') # set the background color of unused cells to this color.
    plt.yticks(rotation=0) # make the y-tick labels straight instead of horizontal.
    plt.ylabel("Count category",labelpad=10)
    plt.xlabel("Count category")
    plt.savefig("../figs/heatmap_aucs.pdf",bbox_inches='tight')
    plt.close()


def main():

    usecols = [2,3,4,5,6,7,8,9,10,11] # ignores CellID (0) and Max (1) columns.
    D = pd.read_csv("../data/mbon_a3/220123_MBONap3_MCHres.csv",usecols=usecols)
    D = D.to_numpy() # convert to numpy.

    assert D.shape == (NUM_CELLS,NUM_TRIALS) # 72 cells x 10 trials.

    fig1(D)  # raw data, medians, exp fit.
    fig3(D)  # kde plot.
    fig4(D)  # heatmap of p-values.
    fig7(D)  # ROC curves: pairwise. (remember to change fig size to (16,10))


if __name__ == "__main__":
    main()
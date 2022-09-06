#!/usr/bin/env python

# © Saket Navlakha, Cold Spring Harbor Laboratory
# Released: September 6, 2022.
# Please cite: Dasgupta, Hattori, Navlakha. Nature Communications, 2022. 

from matplotlib import pylab as plt
from matplotlib import rcParams
from sklearn.preprocessing import normalize
from optparse import OptionParser
import numpy as np
import scipy.stats


# Set plotting style.
plt.style.use('ggplot')
plt.rc('font', **{'sans-serif':'Arial', 'family':'sans-serif'})
rcParams.update({'font.size': 20})


np.random.seed(55)

# Set on the command line or in main().
n=-1        # number of unique inputs.
d=-1        # number of input dimensions.
m=-1        # number of KCs.
max_corr=-1 # max correlation allowed between any pair of inputs.


# Constants.
TOPK=10   # number of active KCs in the tag (i.e., number of hash functions).
MEAN=10   # adjusted mean of each input.
ZIPF=0.55 # parameter for the Zipf distribution.
SUPP=0.44 # suppression constant. 


#==============================================================================
#                                 DATA FUNCTIONS
#==============================================================================
def read_random_data(n,d):
    """ Creates n random vectors drawn from an exponential each of dimension d. """

    X = np.random.exponential(scale=MEAN,size=(n,d))
    
    return standardize_data(X,n,d,remove_neg=False,l2_norm=False)


def read_hallem_data(n,d,filename="data/halem/hallem1.txt"):
    """ Reads the Hallem data. """

    D = np.zeros((n,d))
    with open(filename) as f:
        for line_num,line in enumerate(f):
            if line_num == 0:  continue  # Header.
            if line_num == n+1: continue # Ignore last row, baselines.

            cols = line.strip().split()
            assert len(cols) == d+1

            D[line_num-1,:] = list(map(int,cols[1:]))

    assert line_num == n+1

    return standardize_data(D,n,d,remove_neg=True,l2_norm=False)


def standardize_data(X,n,d,remove_neg=False,l2_norm=False):
    """ Performs several standardizations on the data.
            1) Makes sure all values are non-negative.
            2) Sets the mean of example to MEAN.
            3) Applies l2-normalization if desired.
    """

    # 1. Add the most negative number per column (ORN) to make all values >= 0.
    if remove_neg:
        for col in range(d):
            X[:,col] += abs(min(X[:,col]))


    # 2. Set the mean of each row (odor) to be MEAN.
    for row in range(n):

        # Multiply by: SET_MEAN / current mean. Keeps proportions the same.
        X[row,:] = X[row,:] * ((MEAN / np.mean(X[row,:])))

        assert abs(np.mean(X[row,:]) - MEAN) <= 1


    # 3. Applies normalization.
    if l2_norm: 
        X = normalize(X) # equivalent to: v = v / np.linalg.norm(v)

    # Make sure all values (firing rates) are >= 0.
    if remove_neg:
        for row in range(n):
            for col in range(d):
                assert X[row,col] >= 0

    return X


def read_generic_data(filename,n,d):
    """ Generic reader for: sift, gist, corel, mnist, glove, audio, msong, tsao, fmnist. """

    D = np.zeros((n,d))
    with open(filename) as f:
        for line_num,line in enumerate(f):

            cols = line.strip().split(",")
            assert len(cols) == d

            D[line_num,:] = list(map(float,cols))

            if line_num+1 == n: break # TODO: added for now to make dataset smaller.

    assert line_num+1 == n

    return standardize_data(D,n,d,remove_neg=True,l2_norm=False)


#==============================================================================
#                                 UTILITIES
#==============================================================================
def create_rand_proj_matrix(p=6):
    """ Create sparse, binary random projection matrix of size m x d, with p ones per row (KC). """

    assert p < d  # can't have fewer samples than input dimensions.
    assert p == 6 # fix, each KC samples from 6 glomeruli.

    # Create a sparse, binary random projection matrix.
    # Each row (KC) samples from p random input dimensions.
    M = np.zeros((m,d))
    for row in range(m):

        # Sample p random indices; set these to 1.
        for idx in np.random.choice(range(d),size=p,replace=False):       
            M[row,idx] = 1

        # Make sure I didn't screw anything up!
        assert sum(M[row,:]) == p

    return M


def remove_correlated(X):
    """ Filters X to only include uncorrelated inputs. Picks a random input; if it's 
        correlated with a chosen input, then discard it. Else, it's added to the chosen set. 
        Repeat until nothing else can be added.
    """

    corr = np.corrcoef(X) # matrix of all pairwise correlations.

    indices = [np.random.randint(0,n)] # pick a random input.

    # Iterate through X and add uncorrelated inputs.
    for _ in range(X.shape[0]*10):
        new = np.random.randint(0,n) # candidate.

        # Keep if it's uncorrelated with anything already chosen.
        keep = True
        for i in indices:
            if corr[i][new] >= max_corr: 
                keep = False
                break

        if keep: indices.append(new)

    # Check for no duplicates.
    assert len(indices) == len(set(indices))

    # Return the new n and X.
    return len(indices),X[indices]


def random_zipf():
    """ Returns samples drawn from a Zipf distribution: prob ~ i^-a, where a is the 
        distribution parameter and i is the item. 
        From: https://stackoverflow.com/questions/33331087/sampling-from-a-bounded-domain-zipf-distribution
    """
    
    x = np.arange(1,n+1)     # list of items, from 1-n+1 (can't be 0-n bc then item 0 will have prob 0).
    weights = x ** (-ZIPF)   # generate the zipf distribution values.
    weights /= weights.sum() # convert to probabilities.

    bounded_zipf = scipy.stats.rv_discrete(name='bounded_zipf', values=(x, weights))
    sample = bounded_zipf.rvs(size=n) # sample items.

    return sample - 1 # go back to 0-n indexing.


def random_uniform():
    """ Returns samples from from a uniform distribution. """
    sample = [np.random.randint(0,n) for _ in range(n)]
        
    return sample


#==============================================================================
#                                      MAIN
#==============================================================================
def main():

    global n,d,m,max_corr

    usage="usage: %prog [options]"
    parser = OptionParser(usage=usage)
    parser.add_option("-d", "--dataset", action="store", type="string", dest="dataset", default="random",help="dataset (odors, random)")
    parser.add_option("-x", "--noise", action="store", type="float", dest="noise", default=0,help="noise level")
    parser.add_option("-a", "--alg", action="store", type="string", dest="alg", default="add",help="add/div for weight updates")
    parser.add_option("-t", "--dist", action="store", type="string", dest="dist", default="uniform",help="stream distribution")

    (options, args) = parser.parse_args()

    alg   = options.alg    
    dist  = options.dist
    noise = float(options.noise)
    

    # ===============================================================
    # Read Hallem odors data: 110 odors x 24 ORNs.
    if options.dataset == "odors":
        m = 10000
        n = 110
        d = 24
        max_corr = 0.80
        X = read_hallem_data(n,d)

    # Read Random data: 1000 random exponentials x 50 dimensions.
    elif options.dataset == "random":
        m = 10000
        n = 1000
        d = 50
        max_corr = 0.80
        X = read_random_data(n,d)

    # Read MNIST data: 10000 images x 784 dimensions.
    elif options.dataset == "mnist":
        m = 10000
        n = 10000
        d = 84
        max_corr = 0.70
        X = read_generic_data("data/mnist/mnist_mnist_test.csv",n,d)
        np.random.shuffle(X) # in case the images are ordered by class label.

    else: assert False


    # Remove correlated inputs.
    n,X = remove_correlated(X)
    
    assert X.shape[0] == n
    assert X.shape[1] == d

    
    # ===============================================================

    # Create random projection matrix.
    M = create_rand_proj_matrix()
    Mt = np.transpose(M)
    assert M.shape[0] == m
    assert M.shape[1] == d


    # Create Kenyon cell activities for each input.
    Y = np.dot(X,Mt)
    assert Y.shape[0] == n
    assert Y.shape[1] == m


    # Perform WTA on the KC activities.
    Z = {} # Active KCs for each input: i -> [active KCs]
    for i in range(n):

        # Apply the WTA to Y.
        indices = np.argpartition(Y[i,:],-TOPK)[-TOPK:]    
        assert len(indices) == TOPK

        # Store the active KCs in Z.
        Z[i] = indices


    # Initialize KC->MBON weights.
    if   alg == "add": W = np.zeros((m)) # for additive increase.
    elif alg == "div": W = np.ones((m))  # for multiplicative decrease; 0-1-2-many.
    else: assert False


    # Generate training data.
    if   dist == "zipf"   : training_data = random_zipf()
    elif dist == "uniform": training_data = random_uniform()
    else: assert False


    # =================== TRAINING ===================
    #   Add noise to training data -- removed (see v5).
    #   Add weight recovery -- removed (see v5).

    # Observe items and store them in the sketch.
    Counts = {} # ground truth: item -> count.
    for x in training_data: # comment

        assert 0 <= x <= n-1

        # Get the active KCs.
        indices = Z[x]

        # Update the weights.
        if   alg == "add": W[indices] = W[indices] + 1/TOPK # additive increase.
        elif alg == "div": W[indices] = W[indices] * SUPP   # multiplicative decrease.
        
        # Store the ground-truth.
        Counts[x] = Counts.get(x,0) + 1
    # ================================================


    # =================== TESHTING ===================

    # Test correctness for each item.
    if alg == "add":
        raw_actual,raw_predict = [],[] # to make it easy to compute correlation coef.
        xyplot = {} # actual -> [list of predicted counts] for averages.
    elif alg == "div":
        zero,one,two,many = [],[],[],[] # average responses for each count category.

    for i in range(n):

        # The true counts for i.
        actual = Counts.get(i,0) # might be zero if never drawn.

        if noise == 0:
            indices = Z[i] 
        else:
            assert noise > 0

            # Multiply each dimension by a random number in [3/4,5/4]: Sanjoy's idea.
            noisy_Xi = X[i,:] * np.random.uniform(low=1-noise, high=1+noise, size=d)

            # Calculate KC activities and topk indices for the noisy input.
            noisy_Yi = np.dot(noisy_Xi,Mt)
            indices  = np.argpartition(noisy_Yi,-TOPK)[-TOPK:]
            

        # The predicted counts for i: sum of the active KC->MBON weights.
        predict = np.sum(W[indices])
        
        assert predict >= 0

        # Store numbers for mean+/-std plot.
        if alg == "add":
            xyplot.setdefault(actual,[]).append(predict)
            raw_actual.append(actual)
            raw_predict.append(predict)            

        # Store numbers for zero,one,two,many plot.
        elif alg == "div":
            predict = predict / TOPK # normalize so max response is 1.
            if   actual == 0: zero.append(predict)
            elif actual == 1: one.append(predict)
            elif actual == 2: two.append(predict)
            elif actual > 2:  many.append(predict)
            else: assert False           
    # ================================================


    # =================== PLOTTING ===================
    # Format data for plotting.
    if alg == "add":

        # Correlation between actual and predicted.
        r = np.corrcoef(raw_actual,raw_predict)[0][1]
        label = "r = %.3f" %(r)
    
        # Plot.
        x = xyplot.keys()                        # [unique actual counts]
        y = list(map(np.mean,xyplot.values()))   # [mean of predicted counts]
        yerr = list(map(np.std,xyplot.values())) # [std of predicted counts]

        plt.errorbar(x,y,fmt='o',barsabove=False,linewidth=2,yerr=yerr,label=label,alpha=0.8)

        plt.legend(frameon=True,framealpha=0.6,borderpad=0.25)
        plt.xlabel("True count")
        plt.ylabel("Decoder output")
        #plt.scatter(raw_actual,raw_predict,c='k')

        # Set the range of the x and y ticks to be the same.
        
        if options.dataset == "random":
            ticks = np.arange(0, int(max(max(x),max(y)))+2, 1.0) # range is 0 to max(xmax,ymax)
            tickslabel = [int(i) if i % 3 == 0 else "" for i in ticks] # labels on every nth tick.
        elif options.dataset == "odors":
            ticks = np.arange(0, 8, 1.0) # range is 0 to max(xmax,ymax) # TODO be careful if params change.
            tickslabel = [int(i) if i % 1 == 0 else "" for i in ticks] # labels on every nth tick.
        elif options.dataset == "mnist":
            ticks = np.arange(0, 14, 1.0) # range is 0 to max(xmax,ymax) # TODO be careful if params change.
            tickslabel = [int(i) if i % 3 == 0 else "" for i in ticks] # labels on every nth tick.

        plt.xticks(ticks=ticks,labels=tickslabel)
        plt.yticks(ticks=ticks,labels=tickslabel)

        plt.xlim(min(ticks)-0.5,max(ticks))
        plt.ylim(min(ticks)-0.5,max(ticks))

        # Plot y=x line to see the biased estimator (if no noise) and deviation from perfect.
        xy = [i for i in ticks]
        plt.plot(xy,xy,"--",color='k',alpha=0.4)

        
    elif alg == "div":

        # Output mean/std statistics.
        for i,category in enumerate([zero,one,two,many]):
            print("%i: n=%i, %.3f +/- %.3f" %(i,len(category),np.mean(category),np.std(category)))

    
        # Plot boxplot.
        plt.boxplot([zero,one,two,many],showmeans=True,showfliers=True,labels=["1","2","3","many"])
        plt.xlabel("Count category")
        plt.ylabel("Decoder output")
        plt.ylim(-0.05,1.05)

        # Plot p-vals.
        categories = [zero,one,two,many]
        PFONT = 14
        for i in range(3):
            pval = scipy.stats.ranksums(categories[i],categories[i+1])[1] # p-val of n vs n+1.     

            # Plot significant pvals as red + bold.
            if pval < 0.01:
                plt.text(i+1.5,1.07,"p=%0.1e" %(pval),color="red",fontweight="bold",horizontalalignment="center",fontsize=PFONT,usetex=True)
            else:
                plt.text(i+1.5,1.07,"p=%0.1e" %(pval),alpha=0.5,horizontalalignment="center",fontsize=PFONT,usetex=True)

    # Save fig.
    #plt.title("n = %i, m = %i, topk = %i, dist = %s, noise = %.2f" %(n,m,topk,dist,noise),fontsize=12)    
    plt.savefig("figs/%s_%s_%s_%.2f_%i.pdf" %(options.dataset,alg,dist,noise,m),bbox_inches='tight')
    plt.close()
    plt.show()


    # Output p-vals and additional statistics.
    if alg == "add": 
        print("r=%.3f" %(r))
        used = sum(W!=0) # number of used synapses.

    if alg == "div":
        # Output p-vals (Wilcoxon rank sum).
        print("0 vs 1:", scipy.stats.ranksums(zero,one)[1])
        print("1 vs 2:", scipy.stats.ranksums(one,two)[1])
        print("2 vs 3:", scipy.stats.ranksums(two,many)[1])        

        used = sum(W!=1) # number of synapses used.

    unique = len(set(training_data)) # number of unique inputs in training data.
    frac_overlap = (unique*TOPK - used)/used # (ideal non-overlap - used) / used synapses.

    print("%s: n=%i, d=%i, m=%i, topk=%i, noise=%.2f, dist=%s, used=%i, over=%.2f, unq=%i" %(options.dataset,n,d,m,TOPK,noise,dist,used,frac_overlap,unique))


if __name__ == "__main__":
    main()

# Base Python
import cPickle, json, os, sys, warnings
from collections import defaultdict, OrderedDict

# Scientific modules
import numpy as np
import matplotlib
matplotlib.use('Agg')
warnings.filterwarnings("ignore", module="matplotlib")
import matplotlib.pyplot as plt
import scipy.sparse as sparse


####################################################################
############################ ALGORITHMS ############################
#################################################################### 

"""
We will first consider the *pipelined* approach, consisting of stages:

1. Learn weights theta, parameterizing the generative model of
  training set creation defined by the LFs (and LF dependencies)

2. Learn the weights w of our model (log. reg. here), using
  noise-aware ERM

STAGE 1:

We consider a vector of sufficient statistics for our model
h(X, Y), comprising: 

 Independent factors:
  * x_i * Y     -- E[x_i*Y] is the accuracy
  * x_i^2       -- E[x_i^2] is the coverage
  * x_i
  * x_i^2 * Y
  * Y
 
 Dependency factors:
  * -1{x_i != x_j}                 -- Similar(i,j)
  * 1{x_i*Y == -1 and x_j*Y == 1}  -- Fixes(i,j)
  * 1{x_i*Y == 1 and x_j*Y == 1}   -- Reinforces(i,j)
  * -1{x_i != 0 and x_j != 0}      -- Excludes(i,j)

NOTE: Other formulations that have class-specific acc. possible,
can add this later...

We consider the maximum-entropy family determined by these sufficient
statistics:
P(X,Y;\theta) = Z(\theta)^{-1}\exp(\theta^T h(X,Y))

We will then learn the maximum-likelihood estimator \theta_{MLE}
using SGD:
\grad_theta = E[h(X,Y)|X] - E[h(X,Y)]
"""


def log_odds(p):
  """This is the logit function"""
  return np.log(p / (1.0 - p))

def odds_to_prob(l):
  """
  This is the inverse logit function logit^{-1}:

    l       = \log\frac{p}{1-p}
    \exp(l) = \frac{p}{1-p}
    p       = \frac{\exp(l)}{1 + \exp(l)}
  """
  return np.exp(l) / (1.0 + np.exp(l))

def sample_data(X, w, n_samples):
  """
  Here we do Gibbs sampling over the decision variables (representing our objects), o_j
  corresponding to the columns of X
  The model is just logistic regression, e.g.

    P(o_j=1 | X_{*,j}; w) = logit^{-1}(w \dot X_{*,j})

  This can be calculated exactly, so this is essentially a noisy version of the exact calc...
  """
  N, R = X.shape
  t = np.zeros(N)
  f = np.zeros(N)

  # Take samples of random variables
  idxs = np.round(np.random.rand(n_samples) * (N-1)).astype(int)
  ct = np.bincount(idxs)
  # Estimate probability of correct assignment
  increment = np.random.rand(n_samples) < odds_to_prob(X[idxs, :].dot(w))
  increment_f = -1. * (increment - 1)
  t[idxs] = increment * ct[idxs]
  f[idxs] = increment_f * ct[idxs]
  
  return t, f


def exact_data(X, w, evidence=None):
  """
  We calculate the exact conditional probability of the decision variables in
  logistic regression; see sample_data
  """
  t = odds_to_prob(X.dot(w))
  if evidence is not None:
    t[evidence > 0.0] = 1.0
    t[evidence < 0.0] = 0.0
  return t, 1-t

def abs_sparse(X):
  """ Element-wise absolute value of sparse matrix """
  X_abs = X.copy()
  if sparse.isspmatrix_csr(X) or sparse.isspmatrix_csc(X):
    X_abs.data = np.abs(X_abs.data)
  elif sparse.isspmatrix_lil(X):
    X_abs.data = np.array([np.abs(L) for L in X_abs.data])
  else:
    raise ValueError("Only supports CSR/CSC and LIL matrices")
  return X_abs

def transform_sample_stats(Xt, t, f, Xt_abs=None):
  """
  Here we calculate the expected accuracy of each LF/feature
  (corresponding to the rows of X) wrt to the distribution of samples S:

    E_S[ accuracy_i ] = E_(t,f)[ \frac{TP + TN}{TP + FP + TN + FN} ]
                      = \frac{X_{i|x_{ij}>0}*t - X_{i|x_{ij}<0}*f}{t+f}
                      = \frac12\left(\frac{X*(t-f)}{t+f} + 1\right)
  """
  if Xt_abs is None:
    Xt_abs = abs_sparse(Xt) if sparse.issparse(Xt) else abs(Xt)
  n_pred = Xt_abs.dot(t+f)
  m = (1. / (n_pred + 1e-8)) * (Xt.dot(t) - Xt.dot(f))
  p_correct = (m + 1) / 2
  return p_correct, n_pred

def learn_elasticnet_logreg(X, maxIter=500, tol=1e-6, w0=None, sample=True,
                            n_samples=100, alpha=0, mu_seq=None, n_mu=20,
                            mu_min_ratio=1e-6, rate=0.01, evidence=None,
                            verbose=False):
  """ Perform SGD wrt the weights w
       * w0 is the initial guess for w
       * sample and n_samples determine SGD batch size
       * alpha is the elastic net penalty mixing parameter (0=ridge, 1=lasso)
       * mu is the sequence of elastic net penalties to search over
  """
  if type(X) != np.ndarray and not sparse.issparse(X):
    raise TypeError("Inputs should be np.ndarray type or scipy sparse.")
  N, R = X.shape

  # Pre-generate other matrices
  Xt = X.transpose()
  Xt_abs = abs_sparse(Xt) if sparse.issparse(Xt) else np.abs(Xt)
  
  # Initialize weights if no initial provided
  w0 = np.zeros(R) if w0 is None else w0   
  
  # Check mixing parameter
  if not (0 <= alpha <= 1):
    raise ValueError("Mixing parameter must be in [0,1]")
  
  # Determine penalty parameters  
  if mu_seq is not None:
    mu_seq = np.ravel(mu_seq)
    if not np.all(mu_seq >= 0):
      raise ValueError("Penalty parameters must be non-negative")
    mu_seq.sort()
  else:
    mu_seq = get_mu_seq(n_mu, rate, alpha, mu_min_ratio)

  if evidence is not None:
    evidence = np.ravel(evidence)

  weights = dict()
  # Search over penalty parameter values
  for mu in mu_seq:
    w = w0.copy()
    g = np.zeros(R)
    l = np.zeros(R)
    # Take SGD steps
    for step in range(maxIter):
      if step % 100 == 0 and verbose:    
        if step % 500 == 0:
          print "Learning epoch = ",
        print "%s\t" % step,
        if (step+100) % 500 == 0:
          print "\n",
      
      # Get the expected LF accuracy
      t,f = sample_data(X, w, n_samples=n_samples) if sample else exact_data(X, w, evidence)
      p_correct, n_pred = transform_sample_stats(Xt, t, f, Xt_abs)

      # Get the "empirical log odds"; NB: this assumes one is correct, clamp is for sampling...
      l = np.clip(log_odds(p_correct), -10, 10)

      # SGD step with normalization by the number of samples
      g0 = (n_pred*(w - l)) / np.sum(n_pred)

      # Momentum term for faster training
      g = 0.95*g0 + 0.05*g

      # Check for convergence
      wn = np.linalg.norm(w, ord=2)
      if wn < 1e-12 or np.linalg.norm(g, ord=2) / wn < tol:
        if verbose:
          print "SGD converged for mu={:.3f} after {} steps".format(mu, step)
        break

      # Update weights
      w -= rate*g
      
      # Apply elastic net penalty
      soft = np.abs(w) - alpha * mu
      #          \ell_1 penalty by soft thresholding        |  \ell_2 penalty
      w = (np.sign(w)*np.select([soft>0], [soft], default=0)) / (1+(1-alpha)*mu)
    
    # SGD did not converge    
    else:
      warnings.warn("SGD did not converge for mu={:.3f}. Try increasing maxIter.".format(mu))

    # Store result and set warm start for next penalty
    weights[mu] = w.copy()
    w0 = w
    
  return weights
  
def get_mu_seq(n, rate, alpha, min_ratio):
  mv = (max(float(1 + rate * 10), float(rate * 11)) / (alpha + 1e-3))
  return np.logspace(np.log10(mv * min_ratio), np.log10(mv), n)
  
def cv_elasticnet_logreg(X, nfolds=5, w0=None, mu_seq=None, alpha=0, rate=0.01,
                         mu_min_ratio=1e-6, n_mu=20, opt_1se=True, 
                         verbose=True, plot=True, **kwargs):
  N, R = X.shape
  # Initialize weights if no initial provided
  w0 = np.zeros(R) if w0 is None else w0   
  # Check mixing parameter
  if not (0 <= alpha <= 1):
    raise ValueError("Mixing parameter must be in [0,1]")
  # Determine penalty parameters  
  if mu_seq is not None:
    mu_seq = np.ravel(mu_seq)
    if not np.all(mu_seq >= 0):
      raise ValueError("Penalty parameters must be non-negative")
    mu_seq.sort()
  else:
    mu_seq = get_mu_seq(n_mu, rate, alpha, mu_min_ratio)
  # Partition data
  try:
    folds = np.array_split(np.random.choice(N, N, replace=False), nfolds)
    if len(folds[0]) < 10:
      warnings.warn("Folds are smaller than 10 observations")
  except:
    raise ValueError("Number of folds must be a non-negative integer")
  # Get CV results
  cv_results = defaultdict(lambda : defaultdict(list))
  for nf, test in enumerate(folds):
    if verbose:
      print "Running test fold {}".format(nf)
    train = np.setdiff1d(range(N), test)
    w = learn_elasticnet_logreg(X[train, :], w0=w0, mu_seq=mu_seq, alpha=alpha,
                                rate=rate, verbose=False, **kwargs)
    for mu, wm in w.iteritems():
      spread = 2*np.sqrt(np.mean(np.square(odds_to_prob(X[test,:].dot(wm)) - 0.5)))
      cv_results[mu]['p'].append(spread)
      cv_results[mu]['nnz'].append(np.sum(np.abs(wm) > 1e-12))
  # Average spreads
  p = np.array([np.mean(cv_results[mu]['p']) for mu in mu_seq])
  # Find opt index, sd, and 1 sd rule index
  opt_idx = np.argmax(p)
  p_sd = np.array([np.std(cv_results[mu]['p']) for mu in mu_seq])
  t = np.max(p) - p_sd[opt_idx]
  idx_1se = np.max(np.where(p >= t))
  # Average number of non-zero coefs
  nnzs = np.array([np.mean(cv_results[mu]['nnz']) for mu in mu_seq])
  # glmnet plot
  if plot:
    fig, ax1 = plt.subplots()
    # Plot spread
    ax1.set_xscale('log', nonposx='clip')    
    ax1.scatter(mu_seq[opt_idx], p[opt_idx], marker='*', color='purple', s=500,
                zorder=10, label="Maximum spread: mu={}".format(mu_seq[opt_idx]))
    ax1.scatter(mu_seq[idx_1se], p[idx_1se], marker='*', color='royalblue', 
                s=500, zorder=10, label="1se rule: mu={}".format(mu_seq[idx_1se]))
    ax1.errorbar(mu_seq, p, yerr=p_sd, fmt='ro-', label='Spread statistic')
    ax1.set_xlabel('log(penalty)')
    ax1.set_ylabel('Marginal probability spread: ' + r'$2\sqrt{\mathrm{mean}[(p_i - 0.5)^2]}$')
    ax1.set_ylim(-0.04, 1.04)
    for t1 in ax1.get_yticklabels():
      t1.set_color('r')
    # Plot nnz
    ax2 = ax1.twinx()
    ax2.plot(mu_seq, nnzs, '.--', color='gray', label='Sparsity')
    ax2.set_ylabel('Number of non-zero coefficients')
    ax2.set_ylim(-0.01*np.max(nnzs), np.max(nnzs)*1.01)
    for t2 in ax2.get_yticklabels():
      t2.set_color('gray')
    # Shrink plot for legend
    box1 = ax1.get_position()
    ax1.set_position([box1.x0, box1.y0+box1.height*0.1, box1.width, box1.height*0.9])
    box2 = ax2.get_position()
    ax2.set_position([box2.x0, box2.y0+box2.height*0.1, box2.width, box2.height*0.9])
    plt.title("{}-fold cross validation for elastic net logistic regression with mixing parameter {}".
              format(nfolds, alpha))
    lns1, lbs1 = ax1.get_legend_handles_labels()
    lns2, lbs2 = ax2.get_legend_handles_labels()
    ax1.legend(lns1+lns2, lbs1+lbs2, loc='upper center', bbox_to_anchor=(0.5,-0.05),
               scatterpoints=1, fontsize=10, markerscale=0.5)
    plt.show()
  # Train a model using the 1se mu
  mu_opt = mu_seq[idx_1se if opt_1se else opt_idx]
  w_opt = learn_elasticnet_logreg(X, w0=w0, alpha=alpha, rate=rate,
                                  mu_seq=mu_seq, **kwargs)
  return w_opt[mu_opt]

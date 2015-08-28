"""
Usage:
    lda.py <CORPUS>

Options:
    -k --ntopics    Number of topics to use [default: 20].
    -a --alpha      LDA Alpha Parameter [default: 0.3].
    -b --beta       LDA Beta Parameter [default: 0.3].
    -f --format     Format of the CORPUS file [default: pickle].
    -n --iter       Number of sample iterations to perform [default: 2000]
    -b --burn       Number of samples to discard at first [default: 300]
    -t --thin       Interval of iterations to measure [default: 1]

A reference implementation of LDA using Gibbs Sampling.
"""

from docopt import docopt
import numpy as np
import pickle
import pymc as pm
import scipy as sp
from scipy import sparse


def save_sparse(fname, array):
    np.savez(fname, data=array.data, indices=array.indices, indptr=array.indptr,
            shape=array.shape)

def load_csr(fname):
    loader = np.load(fname)
    return sparse.csr_matrix((loader['data'], loader['indices'],
    loader['indptr']), shape=loader['shape'])

def load_csc(fname):
    loader = np.load(fname)
    return sparse.csc_matrix((loader['data'], loader['indices'],
    loader['indptr']), shape=loader['shape'])

def load_pickle(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)

def load(fname, fmt):
    if fmt == "pickle":
        return load_pickle(fname)
    if fmt == "csr":
        return load_csr(fname)
    if fmt == "csc":
        return load_csc(fname)
    raise ValueError("Unknown file format %s" % fmt)


def main(args):

    K = int(args['--ntopics'])
    data = load(args['<CORPUS>'], args['--format'])
    D = data.shape[0]  # number of documents
    V = data.shape[1]  # number of words

    alpha = np.ones(K) * float(args['--alpha'])
    beta = np.ones(V) * float(args['--alpha'])


    theta = pm.Container([
        pm.CompletedDirichlet(
            "theta_%s" % i,
            pm.Dirichlet("ptheta_%s" % i, theta=alpha)
        ) for i in range(D)
    ])
    phi = pm.Container([
        pm.CompletedDirichlet(
            "phi_%s" % k,
            pm.Dirichlet("pphi_%s" % k, theta=beta)
        ) for k in range(K)
    ])
    Wd = [len(doc) for doc in data]

    z = pm.Container([pm.Categorical('z_%i' % d,
                         p = theta[d],
                         size=Wd[d],
                         value=np.random.randint(K, size=Wd[d]))
                      for d in range(D)])

    # cannot use p=phi[z[d][i]] here since phi is an ordinary list while z[d][i] is stochastic
    w = pm.Container([pm.Categorical("w_%i_%i" % (d,i),
                        p = pm.Lambda('phi_z_%i_%i' % (d,i),
                                  lambda z=z[d][i], phi=phi: phi[z]),
                        value=data[d][i],
                        observed=True)
                      for d in range(D) for i in range(Wd[d])])

    model = pm.Model([theta, phi, z, w])
    mcmc = pm.MCMC(model)
    niter = int(args['--iter'])
    burn = int(args['--burn'])
    thin = int(args['--thin'])
    mcmc.sample(niter, burn, thin)

if __name__ == "__main__":
    arguments = docopt(__doc__, version="lda.py 0.1.0")
    print("ARGS: %s" % arguments)
    main(arguments)

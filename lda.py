"""
Usage:
    lda.py <DOCUMENTS>

Options:
    -k --ntopics    Number of topics to use [default: 20].
    -a --alpha      LDA Alpha Parameter [default: 0.3].
    -b --beta       LDA Beta Parameter [default: 0.3].

A reference implementation of LDA using Gibbs Sampling.  <DOCUMENTS> should be
a file where each line is a document, represented by a unique identifier
followed by word counds in the form of WORD:COUNT, separated by spaces.
"""

import numpy as np
import pymc as pm
from docopt import docopt

class Corpus(object):
    """
    A class encapsulating a document corpus
    """

    def __init__(self, doc_stream):
        """
        Construct a new corpus object from a text stream
        """
        self.doc_ids = []
        self.doc_word_counts = []
        self.vocab = set()
        for doc_idx, line in enumerate(doc_stream):
            line = line.strip().split()
            doc_id = line[0]
            self.doc_ids.append(doc_id)



def main(args):

    K = 2  # number of topics
    V = 4  # number of words
    D = 3  # number of documents

    data = np.array([[1, 1, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0]])

    alpha = np.ones(K)
    beta = np.ones(V)

    theta = pm.Container([pm.CompletedDirichlet("theta_%s" % i, pm.Dirichlet("ptheta_%s" % i, theta=alpha)) for i in range(D)])
    phi = pm.Container([pm.CompletedDirichlet("phi_%s" % k, pm.Dirichlet("pphi_%s" % k, theta=beta)) for k in range(K)])
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
    mcmc.sample(100)

if __name__ == "__main__":
    arguments = docopt(__doc__, version="lda.py 0.1.0")
    main(arguments)

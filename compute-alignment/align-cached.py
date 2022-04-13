# coding: utf-8
# Copyright (c) 2018 - Present Bill Thompson (billdthompson@berkeley.edu)  

import numpy as np
import pandas as pd
import wordfreq as wf
import editdistance as ed
from scipy.stats import pearsonr
import click

import logging
logging.basicConfig(format='%(levelname)s > %(message)s', level=logging.INFO)

AL = wf.available_languages()

def freq(w ,l):
    """get log frequency: w = wordform; l = iso_2"""
    if l in AL:
        try: 
            zf = wf.zipf_frequency(w, l)
            return zf if zf > 0 else None
        except:
            return None
    else:
        return None

def vectors(v):
    return pd.read_csv(v, header = None).values

def network(v, w):
    """return cosine similarity network: v = vecfilepath, w = wordlist series"""
    V = vectors(v)

    # normalise vectors to unit length (division by euclidian norm)
    unitvectors = V / np.sqrt((V ** 2).sum(axis = 1, keepdims = True))

    # for unit vectors, cosine similarity = dot product
    return np.dot(unitvectors, unitvectors.T)

def nearest_k(A, K):
    """Identify nearest K neighbours
    A: N x N ndarray pairwise cosine sims (network);
    K: How many neighbours per meaning?; 
    """
    N = A.shape[0]
    X = np.zeros_like(A).astype(bool)
    
    # flip boolen for row-wise top K neighbours 
    X[np.arange(N).repeat(K), np.argpartition(A, -K)[:,-K:].flatten()] = True
    return X

def align(A, B, X, Y):
    """compute alignment:
    A: N x N ndarray pairwise cosine sims in l1 (network);
    B: N x N ndarray pairwise cosine sims in l2 (network);
    X: N x N ndarray of boolean == True if X_{i,j} is considered a neighbour in l1;
    Y: N x N ndarray of boolean == True if Y_{i,j} is considered a neighbour in l2;
    """
    N = A.shape[0]

    # full row alignment
    global_alignment = np.einsum('ij,ij->i', A / np.sqrt((A ** 2).sum(axis = 1, keepdims = True)), B / np.sqrt((B ** 2).sum(axis = 1, keepdims = True)))

    # (two-sided) neighbour alignment
    local_alignment = np.ones(N)
    with click.progressbar(np.arange(N), label='Computing Alignment Correlation', length=N) as bar:
        for i in bar:
            r1, r2 = pearsonr(A[i][X[i]], B[i][X[i]])[0], pearsonr(B[i][Y[i]], A[i][Y[i]])[0]
            local_alignment[i] = np.mean([r1, r2])
    return local_alignment, global_alignment

@click.command()
@click.option('--vectorfile1', '-v1')
@click.option('--vectorfile2', '-v2') 
@click.option('--wordlistfile', '-w') 
@click.option('--l1', '-l1')   
@click.option('--l2', '-l2')   
@click.option('--k', '-k', default = 40) 
def run(vectorfile1, vectorfile2, l1, l2, k, wordlistfile):
    click.secho("ALIGN > Computing alignment: {0} & {1}".format(l1, l2), fg='green')
    wordlist = pd.read_csv(wordlistfile)

    logging.info("Computing semantic network in: {0}".format(l1))
    netl1 = network(vectorfile1, wordlist[l1].str.lower())

    logging.info("Computing semantic network in: {0}".format(l2))
    netl2 = network(vectorfile2, wordlist[l2].str.lower())
    
    # top K neighbours
    # = K + 1 because we'll cut out the diagonal afterwards
    logging.info("Computing nearest {} neighbours for all {} words in both languages.".format(k, wordlist.shape[0]))
    kL1, kL2 = nearest_k(netl1, k + 1), nearest_k(netl2, k + 1)
    np.fill_diagonal(kL1, False), np.fill_diagonal(kL2, False)

    # gather results
    results = pd.DataFrame(dict(l1 = l1, l2 = l2, wordform_l1 = wordlist[l1].str.lower(), wordform_l2 = wordlist[l2].str.lower()))
    results['local_alignment'],  results['global_alignment'] = align(netl1, netl2, kL1, kL2)
    results['freq_l1'] = results.wordform_l1.apply(lambda wf: freq(wf, l1) or None)
    results['freq_l2'] = results.wordform_l2.apply(lambda wf: freq(wf, l2) or None)
    results['neighbour_overlap'] = (kL1 & kL2).sum(axis = 1).astype(float)
    results['global_density_l1'] = netl1.mean(axis = 1) 
    results['global_density_l2'] = netl2.mean(axis = 1) 
    results['local_density_l1'] = np.array(np.ma.array(netl1, mask = ~kL1)).sum(axis = 1) 
    results['local_density_l2'] = np.array(np.ma.array(netl2, mask = ~kL2)).sum(axis = 1)
    results['editdistance'] = results.apply(lambda row: ed.eval(row.wordform_l1, row.wordform_l2), axis = 1)
    results['k'] = k
    results['n'] = wordlist.shape[0]

    # save results
    filename = '--'.join(["w=" + wordlistfile.split('/')[-1].replace('.csv', ''), "v1=" + vectorfile1.split('/')[-1], "v2=" + vectorfile2.split('/')[-1], "k=" + str(k), "N=" + str(wordlist.shape[0])]) + '.csv'
    logging.info("saving results to: {}".format(filename))
    results.sort_values('local_alignment').to_csv(filename, index = False)

if __name__ == '__main__':
    run()









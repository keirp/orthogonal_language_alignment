# coding: utf-8
# Copyright (c) 2018 - Present Bill Thompson (billdthompson@berkeley.edu)  

import numpy as np
import pandas as pd
import click

import logging
logging.basicConfig(format='%(levelname)s > %(message)s', level=logging.INFO)

D = 300 # skipgram vector dimension
VECTORDIMENSIONS = ['d_{0}'.format(d) for d in range(D)]

def vocab(v):
	"""return vocabulary list: v = vecfilepath"""
	return np.array([line.split(' ')[0] for line in open(v, 'r')]) 

def vectors(v, w):
    """return ordered target vectors: v = vecfilepath, w = target word array"""
    data = np.array([np.array(line.rstrip('\n').split(' ')) for line in open(v, 'r') if line.split(' ')[0] in w])
    vectors = pd.DataFrame(dict(zip(VECTORDIMENSIONS, data[:,1:].T)), dtype = np.float)
    vectors['word'] = data.T[0]
    return vectors.set_index('word').loc[w].reset_index()

@click.command()
@click.option('--vectorfile1', '-v1') # [...].l1.vec
@click.option('--vectorfile2', '-v2') # [...].l2.vec
@click.option('--l1', '-l1')   # iso_2 e.g. en
@click.option('--l2', '-l2')   # iso_2 e.g. de
@click.option('--translations', '-t') # translation dictionary
@click.option('--ntrans', '-n', default = 20000) # max number of translations to use
def run(vectorfile1, vectorfile2, l1, l2, translations, ntrans):
	# handy variable renames
	click.secho("Cacheing matched-order vectors for: {0} & {1}".format(l1, l2), fg='green')
	wordlist = pd.read_csv(translations, nrows = ntrans)

	logging.info("Formatting {}.".format(translations))
	wordlist[l1] = wordlist[l1].str.lower()
	wordlist[l2] = wordlist[l2].str.lower()
	wordlist[l1] = wordlist[l1].str.replace(' ', '_')
	wordlist[l2] = wordlist[l2].str.replace(' ', '_')

	logging.info("Obtained {0} translation pairs in total.".format(wordlist.shape[0]))
	logging.info("Collecting vector vocabularies.")
	vocabl1 = vocab(vectorfile1)
	logging.info("{0} contains vectors for {1} words".format(vectorfile1, len(vocabl1)))
	
	vocabl2 = vocab(vectorfile2)
	logging.info("{0} contains vectors for {1} words".format(vectorfile2, len(vocabl2)))

	logging.info("Subsetting translations by common vector coverage.")
	wordlist = wordlist[(wordlist[l1].isin(vocabl1)) & (wordlist[l2].isin(vocabl2))].drop_duplicates(subset = [l1, l2])
	logging.info("Established common vector coverage for {} uique translation pairs".format(wordlist.shape[0]))

	logging.info("Collecting L1 vectors.")
	vecsl1 = vectors(vectorfile1, wordlist[l1].values).rename(columns = {'word':l1})
	logging.info("Collecting L2 vectors.")
	vecsl2 = vectors(vectorfile2, wordlist[l2].values).rename(columns = {'word':l2})
	assert not vecsl1.isnull().any(axis = 1).any() # check no missing vectors
	assert not vecsl2.isnull().any(axis = 1).any() # check no missing vectors

	fn1 = '{}-{}.v1'.format(l1, l2)
	fn2 = '{}-{}.v2'.format(l1, l2)
	wordlistfn = '{}-{}.wordlist'.format(l1, l2)
	logging.info("Saving translation set to: {}".format(wordlistfn))
	wordlist.to_csv(wordlistfn)

	logging.info("Cacheing ordered vectors for {0} to: {1} and {2}".format(wordlistfn, fn1, fn2))
	vecsl1[VECTORDIMENSIONS].to_csv(fn1, header = False, index = False)
	vecsl2[VECTORDIMENSIONS].to_csv(fn2, header = False, index = False)

if __name__ == '__main__':
	run()









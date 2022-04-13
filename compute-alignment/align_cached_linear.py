# coding: utf-8
# Copyright (c) 2018 - Present Bill Thompson (billdthompson@berkeley.edu)  

import numpy as np
import pandas as pd
import wordfreq as wf
import editdistance as ed
from scipy.stats import pearsonr
import click
import scipy

import torch
from torch import nn
import torch.optim as optim

import os
from tqdm import tqdm

os.environ['KMP_DUPLICATE_LIB_OK']='True'

import logging
logging.basicConfig(format='%(levelname)s > %(message)s', level=logging.INFO)

AL = wf.available_languages()

def vectors(v):
	return pd.read_csv(v, header = None).values

def get_alignment(A, B, n_iter):
	learning_rate = 0.0001
	batch_size = 200

	n, d = A.shape

	# model = nn.Sequential(nn.Linear(d, 256),
 #                          nn.BatchNorm1d(256),
 #                          nn.ReLU(),
 #                          nn.Linear(256, 256),
 #                          nn.BatchNorm1d(256),
 #                          nn.ReLU(),
 #                          nn.Linear(256, d))

	model = nn.Linear(d, d, bias=False)

	def loss_fn(y, pred_y):
		return ((y - pred_y) ** 2).mean()

	optimizer = optim.Adam(model.parameters(), lr=learning_rate)

	for i in tqdm(range(n_iter)):
		model.train()
		idx = np.random.choice(n, size=batch_size)
		x = torch.tensor(A[idx]).float()
		y = torch.tensor(B[idx]).float()

		optimizer.zero_grad()
		y_pred = model(x)
		loss = loss_fn(y, y_pred)
		loss.backward()
		optimizer.step()

		if i % 3000 == 0:
			model.eval()
			tqdm.write(f'Iteration {i}:')
			tqdm.write(f'  Training loss: {loss.item():0.2f}')

	model.eval()
	return lambda x: model(torch.tensor(x.reshape(-1, d)).float()).detach().cpu().numpy()

@click.command()
@click.option('--vectorfile1', '-v1')
@click.option('--vectorfile2', '-v2') 
@click.option('--wordlistfile', '-w') 
@click.option('--l1', '-l1')   
@click.option('--l2', '-l2')   
def run(vectorfile1, vectorfile2, l1, l2, wordlistfile):
	click.secho("ALIGN > Computing alignment: {0} & {1}".format(l1, l2), fg='green')
	wordlist = pd.read_csv(wordlistfile)

	A = np.array(vectors(vectorfile1))
	B = np.array(vectors(vectorfile2))

	A_mean = A.mean(axis=0)
	A_std = A.std(axis=0) + 0.001

	B_mean = B.mean(axis=0)
	B_std = B.std(axis=0) + 0.001

	A = A / np.linalg.norm(A, axis=1, keepdims=True)
	B = B / np.linalg.norm(B, axis=1, keepdims=True)

	# A = (A - A_mean) / A_std
	# B = (B - B_mean) / B_std  

	M = B.T @ A
	U, S, VT = scipy.linalg.svd(M)

	W = U @ VT


	proj_A = A @ W.T

	# aligner = get_alignment(A, B, n_iter=50000)
	# proj_A = aligner(A)

	errors = -((proj_A - B) ** 2).mean(axis=-1)

	print(f'Total error: {errors.mean()}')

	results = pd.DataFrame(dict(l1 = l1, l2 = l2, wordform_l1 = wordlist[l1].str.lower(), wordform_l2 = wordlist[l2].str.lower()))

	results['mapping_error_alignment'] = errors

	filename = '--'.join(["w=" + wordlistfile.split('/')[-1].replace('.csv', ''), "v1=" + vectorfile1.split('/')[-1], "v2=" + vectorfile2.split('/')[-1], "N=" + str(wordlist.shape[0])]) + 'linear.csv'
	logging.info("saving results to: {}".format(filename))
	results.sort_values('mapping_error_alignment').to_csv(filename, index = False)

if __name__ == '__main__':
	run()









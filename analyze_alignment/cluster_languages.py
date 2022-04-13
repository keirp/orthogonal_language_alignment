import click
import csv
from scipy.stats import pearsonr
import pandas as pd
import glob
import numpy as np
from sklearn.cluster import SpectralClustering
import networkx as nx
import matplotlib.pyplot as plt

@click.command()
@click.option('--data_csv', '-d')
def run(data_csv):
	# lin_data = []
	# files = glob.glob('../compute-alignment/w=*')
	all_langs = set()
	# for fn in files:
	# 	with open(fn, 'r') as file:
	# 		langs = fn.replace('../compute-alignment/w=', '').split('.')[0].split('-')
	# 		l1, l2 = langs
	# 		csv_reader = csv.reader(file, delimiter=',')
	# 		csv_reader = list(csv_reader)
	# 		n_rows = len(csv_reader)
	# 		if n_rows > 500:
	# 			for row in csv_reader:
	# 				if row[0] == 'l1':
	# 					continue
	# 				w1 = row[2]
	# 				w2 = row[3]
	# 				all_langs.add(l1)
	# 				all_langs.add(l2)
	# 				alignment = float(row[4])
	# 				lin_data.append(dict(lang_pair=f'{l1}-{l2}', 
	# 					                 l1=l1,
	# 					                 l2=l2,
	# 				                     w1=w1,
	# 				                     w2=w2,
	# 				                     linear_alignment=alignment))

	lin_data = []
	with open(data_csv, 'r') as file:
		csv_reader = csv.reader(file, delimiter=',')
		for row in csv_reader:
			if row[0] == '':
				continue
			l1 = row[1]
			l2 = row[2]
			alignment = float(row[3])
			w1 = row[4]
			w2 = row[5]
			all_langs.add(l1)
			all_langs.add(l2)
			lin_data.append(dict(lang_pair=f'{l1}-{l2}', 
				             l1=l1, 
				             l2=l2, 
				             linear_alignment=alignment, 
				             w1=w1, 
				             w2=w2))

	lang_map = dict()
	with open('data/distances/FAIR_languages_glotto_xdid.csv', 'r') as file:
		csv_reader = csv.reader(file, delimiter=',')
		for row in csv_reader: 
			if row[0] == 'Language':
				continue
			full = row[0]
			short = row[8]
			lang_map[short] = full

	print(all_langs)
	n = len(all_langs)
	all_langs = list(all_langs)

	dists = np.zeros((n, n))

	df = pd.DataFrame(lin_data)

	df2 = df
	df2 = df2[df2['l1']=='en']
	df2 = df2.groupby('lang_pair').mean()
	df2 = df2.sort_values(by='linear_alignment')

	for i, row in df2.iterrows():
		langs = row.name
		alignment = row['linear_alignment']
		l1, l2 = langs.split('-')
		print(lang_map[l2], alignment)


	df = df.groupby('lang_pair').mean()
	df = df.sort_values(by='linear_alignment')

	G = nx.Graph()
	for lang in all_langs:
		G.add_node(lang_map[lang])

	for i, row in df.iterrows():
		langs = row.name
		alignment = row['linear_alignment']
		l1, l2 = langs.split('-')
		i1, i2 = all_langs.index(l1), all_langs.index(l1)
		dists[i1, i2] = alignment
		dists[i2, i1] = alignment
		G.add_edge(lang_map[l1], lang_map[l2], weight=alignment)


	max_val = dists.max()
	dists[dists != 0] = max_val - dists[dists!=0]

	# for i in range(n):
	# 	print(all_langs[i], (dists[i] == 0).sum())

	

	pos = nx.spring_layout(G, k=0.3, iterations=300)

	node_pos = np.array([pos[key] for key in pos])


	n_clusters = 10
	clustering = SpectralClustering(n_clusters=n_clusters, affinity='precomputed').fit(dists)
	print(clustering.labels_)

	nx.draw_networkx_nodes(G, pos=pos, nodelist=G.nodes(), node_color=clustering.labels_, cmap='Pastel2')
	# nx.draw_networkx_edges(G, pos=pos, edgelist = G.edges())
	nx.draw_networkx_labels(G, pos=pos, font_size=4)
	plt.savefig('graph_plot.pdf')

	for cluster in range(n_clusters):
		print(f'Cluster {cluster}:')
		for c, lang in zip(clustering.labels_, all_langs):
			if c == cluster:
				print(f'  {lang_map[lang]}')

if __name__ == '__main__':
	run()
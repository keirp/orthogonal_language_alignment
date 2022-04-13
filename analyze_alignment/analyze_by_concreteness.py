import click
import csv
from scipy.stats import pearsonr
import pandas as pd
import glob

@click.command()
@click.option('--data_csv', '-d')
@click.option('--conc_txt', '-c')
def run(data_csv, conc_txt):
	data = []
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
			domain = row[16]
			data.append(dict(l1=l1, 
				             l2=l2, 
				             alignment=alignment, 
				             w1=w1, 
				             w2=w2, 
				             domain=domain))

	lin_data = []
	files = glob.glob('../compute-alignment/w=*')
	for fn in files:
		with open(fn, 'r') as file:
			langs = fn.replace('../compute-alignment/w=', '').split('.')[0].split('-')
			l1, l2 = langs
			csv_reader = csv.reader(file, delimiter=',')
			for row in csv_reader:
				if row[0] == 'l1':
					continue
				w1 = row[2]
				w2 = row[3]
				alignment = float(row[4])
				lin_data.append(dict(l1=l1, 
				                     l2=l2,
				                     w1=w1,
				                     w2=w2,
				                     linear_alignment=alignment))

	conc_data = []
	with open(conc_txt, 'r') as file:
		csv_reader = csv.reader(file, delimiter='\t')
		for row in csv_reader:
			if row[0] == 'Word':
				continue
			word = row[0]
			conc = row[2]
			conc_data.append(dict(w1=word,
				                  conc=conc,
				                  l1='en'))

	conc_df = pd.DataFrame(conc_data)

	df = pd.DataFrame(data)

	lin_df = pd.DataFrame(lin_data)

	df = pd.merge(df, lin_df, how='left', on=['l1', 'l2', 'w1', 'w2'])
	df = df[df['linear_alignment'].notnull()]

	df = df[df['l1'] == 'en']

	df = df.groupby('w1').mean()

	df = pd.merge(df, conc_df, how='left', on=['w1'])

	df = df[df['conc'].notnull()]

	alignments = df[['alignment']].to_numpy().astype(float).ravel()
	lin_alignments = df[['linear_alignment']].to_numpy().astype(float).ravel()
	concs = df[['conc']].to_numpy().astype(float).ravel()

	alignment_corr = pearsonr(alignments, concs)[0]
	lin_alignment_corr = pearsonr(lin_alignments, concs)[0]

	print(f'Alignment Correlation: {alignment_corr}')
	print(f'Linear Alignment Correlation: {lin_alignment_corr}')

if __name__ == '__main__':
	run()
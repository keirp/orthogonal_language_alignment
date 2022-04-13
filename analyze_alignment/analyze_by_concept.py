import click
import csv
from scipy.stats import pearsonr
import pandas as pd
import glob

@click.command()
@click.option('--data_csv', '-d')
def run(data_csv):
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

	lin_df = pd.DataFrame(lin_data)

	df = pd.DataFrame(data)

	df = pd.merge(df, lin_df, how='left', on=['l1', 'l2', 'w1', 'w2'])
	df = df[df['linear_alignment'].notnull()]
	print(df)
	df = df.groupby('domain').mean()
	df = df.sort_values(by='alignment')
	print(df)

	df = df.sort_values(by='linear_alignment')
	print(df)

if __name__ == '__main__':
	run()
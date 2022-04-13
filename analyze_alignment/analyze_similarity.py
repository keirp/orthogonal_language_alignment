import click
import csv
from scipy.stats import pearsonr

@click.command()
@click.option('--pred_align', '-p') # [...].l1.vec
@click.option('--human_align', '-h') # [...].l2.vec
def run(pred_align, human_align):
	pred_data = []
	with open(pred_align, 'r') as file:
		csv_reader = csv.reader(file, delimiter=',')
		for row in csv_reader:
			if row[0] == 'l1':
				continue
			en_word = row[2]
			ja_word = row[3]
			alignment = float(row[4])
			pred_data.append((en_word, ja_word, alignment))
	human_data = []
	with open(human_align, 'r') as file:
		csv_reader = csv.reader(file, delimiter=',')
		for row in csv_reader:
			if row[1] == 'l1':
				continue
			alignment = row[-1]
			en_word = row[3]
			ja_word = row[4]
			if alignment != '':
				human_data.append((en_word, ja_word, float(alignment)))
	
	aligned_data = []
	for pred in pred_data:
		for human in human_data:
			en1, ja1, ali1 = pred
			en2, ja2, ali2 = human
			if en1 == en2 and ja1 == ja2:
				aligned_data.append((en1, ja1, ali1, ali2))

	preds = [x[2] for x in aligned_data]
	humans = [x[3] for x in aligned_data]
	print(len(preds))


	corr = pearsonr(preds, humans)[0]
	print(f'Correlation: {corr}')


if __name__ == '__main__':
	run()
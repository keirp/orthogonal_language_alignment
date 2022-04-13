import click
import glob
import os

@click.command()
def run():
	files = glob.glob('nel-translation-pairs/*')
	lang_files = glob.glob('*.v1')
	existing_langs = [x.split('.')[0] for x in lang_files]
	for i, translation_file in enumerate(files):
		langs = translation_file.split('/')[-1].split('.')[0].split('-')
		if '-'.join(langs) not in existing_langs:
			vec_files = [f'nel-vectors/wiki.nel.{lang}.vec' for lang in langs]

			os.system(f'python cache-vecs.py -v1 {vec_files[0]} -v2 {vec_files[1]} -l1 {langs[0]} -l2 {langs[1]} -t {translation_file}')

			file_name = f'{langs[0]}-{langs[1]}'
			os.system(f'python align_cached_linear.py -v1 {file_name}.v1 -v2 {file_name}.v2 -w {file_name}.wordlist -l1 {langs[0]} -l2 {langs[1]}')

			print(f'Completed {i} out of {len(files)} alignment computations...')

if __name__ == '__main__':
	run()
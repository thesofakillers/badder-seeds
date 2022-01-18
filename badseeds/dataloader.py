import os
import torchtext
import gzip
import shutil
import gdown


def download_and_unzip(destination, out_file_path, file_id):
	name = destination.split('.')[-3]
	if not os.path.isfile(destination):
		url = 'https://drive.google.com/uc?id=' + file_id
		gdown.download(url, destination, quiet=False)
	
	if not os.path.isfile(out_file_path):
		print(f"Unzipping the {name} dataset")
		with gzip.open(destination, 'rb') as f_in:
			with open(out_file_path, 'wb') as f_out:
				shutil.copyfileobj(f_in, f_out)
		print(f'Finished unzipping the {name} dataset')

if __name__ == "__main__":
	#Create folder for data
	if not os.path.isdir('../data'):
		os.makedirs('../data')
		print("Created folder : ", '../data')

	#Download the WikiText dataset
	if not os.path.isdir('../data/WikiText103'):
		print("Downloading the WikiText103 dataset")
		train, valid, test = torchtext.datasets.WikiText103(root='../data', split=('train', 'valid', 'test'))
		print('Finished downloading the WikiText103 dataset')

	#Download the GoodReads dataset
	destination = '../data/goodreads_reviews_dedup.json.gz'
	out_file_path = '../data/goodreads_reviews_dedup.json'
	file_id = '1pQnXa7DWLdeUpvUFsKusYzwbA5CAAZx7'
	download_and_unzip(destination, out_file_path, file_id)

	#Download the GoogleNews dataset
	destination = '../data/GoogleNews-vectors-negative300.bin.gz'
	out_file_path = '../data/GoogleNews-vectors-negative300.bin'
	file_id = '0B7XkCwpI5KDYNlNUTTlSS21pQmM'
	download_and_unzip(destination, out_file_path, file_id)

	#Download the NYT dataset
	destination = '../data/nytimes_news_articles.txt.gz'
	out_file_path = '../data/nytimes_news_articles.txt'
	file_id = '1ITZ6FZq4_C2hs7k540ZYiReNTlWGt4nz'
	download_and_unzip(destination, out_file_path, file_id)
	
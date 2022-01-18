import numpy as np
import os
import torch
import torchtext
import requests
import gzip
import json
import shutil
import binascii


def download_file_from_google_drive(id, destination):
	URL = "https://docs.google.com/uc?export=download"

	session = requests.Session()

	response = session.get(URL, params = { 'id' : id }, stream = True)
	token = get_confirm_token(response)

	if token:
		params = { 'id' : id, 'confirm' : token }
		response = session.get(URL, params = params, stream = True)

	save_response_content(response, destination)    

def get_confirm_token(response):
	for key, value in response.cookies.items():
		if key.startswith('download_warning'):
			return value

	return None

def save_response_content(response, destination):
	CHUNK_SIZE = 32768

	with open(destination, "wb") as f:
		for chunk in response.iter_content(CHUNK_SIZE):
			if chunk: # filter out keep-alive new chunks
				f.write(chunk)

def download_and_unzip(destination, out_file_path, file_id):
	name = destination.split('.')[-3]
	if not os.path.isfile(destination):
		print(f"Downloading the {name} dataset")
		download_file_from_google_drive(file_id, destination)
	
	if not os.path.isfile(out_file_path):
		print(f"Unzipping the {name} dataset")
		with gzip.open(destination, 'rb') as f_in:
			with open(out_file_path, 'wb') as f_out:
				shutil.copyfileobj(f_in, f_out)
		print(f'Finished downloading the {name} dataset')

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
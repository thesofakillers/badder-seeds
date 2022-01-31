import argparse
import os
import gzip
import shutil
from zipfile import ZipFile
import requests
import getopt
import sys
import json

import torchtext
import gdown


class LoadTheData:
    def __init__(self, config):
        self.config = config

    def download_and_unzip(self, destination, out_file_path, file_id):
        name = destination.split(".")[-3]
        if not os.path.isfile(destination):
            url = "https://drive.google.com/uc?id=" + file_id
            gdown.download(url, destination, quiet=False)

        if not os.path.isfile(out_file_path):
            print(f"Unzipping the {name} dataset")

            extension = destination.split(".")[-1]
            if extension == "gz":
                with gzip.open(destination, "rb") as f_in:
                    with open(out_file_path, "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)
                print(f"Finished unzipping the {name} dataset")

            elif extension == "zip":
                with ZipFile(destination, "r") as f_in:
                    with open(out_file_path, "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)
                print(f"Finished unzipping the {name} dataset")

            else:
                raise ValueError("Extension not supported yet")


    def unzip_folder(self, destination, out_file_path, file_id):
        name = destination.split(".")[-3]
        if not os.path.isfile(destination):
            url = "https://drive.google.com/uc?id=" + file_id
            gdown.download(url, destination, quiet=False)

        if not os.path.isfile(out_file_path):
            print(f"Unzipping the {name} dataset")

            extension = destination.split(".")[-1]
            with ZipFile(destination, "r") as zip:
                print("Extracting all the files now...")
                zip.extractall(out_file_path)
                print(f"Finished unzipping the {destination} dataset")


    def download_raw(self):
        raw_path = self.config["raw"]["dir_path"]
        # Create folder for data
        if not os.path.isdir(raw_path):
            os.makedirs(raw_path)
            print("Created folder : ", raw_path)
        # Download the WikiText dataset
        wiki_path = os.path.join(raw_path, self.config["raw"]["wiki_subpath"])
        if not os.path.isdir(wiki_path):
            print("Downloading the WikiText103 dataset")
            train, valid, test = torchtext.datasets.WikiText103(
                root=wiki_path, split=("train", "valid", "test")
            )
            print("Finished downloading the WikiText103 dataset")
        # Download the GoodReads History Biography dataset
        gr_hb_path = os.path.join(raw_path, self.config["raw"]["goodreads_hb_subpath"])
        destination = gr_hb_path + ".json.gz"
        out_file_path = gr_hb_path + ".json"
        file_id = "1lDkTzM6zpYU-HGkVAQgsw0dBzik-Zde9"
        self.download_and_unzip(destination, out_file_path, file_id)
        # Download the GoodReads Romance dataset
        gr_r_path = os.path.join(raw_path, self.config["raw"]["goodreads_r_subpath"])
        destination = gr_r_path + ".json.gz"
        out_file_path = gr_r_path + ".json"
        file_id = "1NpFsDQKBj_lrTzSASfyKbmkSykzN88wE"
        self.download_and_unzip(destination, out_file_path, file_id)
        # Download the NYT dataset
        nyt_path = os.path.join(raw_path, self.config["raw"]["nyt_subpath"])
        destination = nyt_path + ".txt.gz"
        out_file_path = nyt_path + ".txt"
        file_id = "1-2LL6wgTwDzTKfPx3RQrXi-LS6lraFYn"
        self.download_and_unzip(destination, out_file_path, file_id)

    def download_seeds(self):
        seeds_path = self.config["seeds"]["dir_path"]
        # Create folder for seed data
        if not os.path.isdir(seeds_path):
            os.makedirs(seeds_path)
            print("Created folder : ", seeds_path)

        # download seeds
        receive = requests.get(
            "https://raw.githubusercontent.com/maria-antoniak/bad-seeds/main/gathered_seeds.json"
        )
        # r_dictionary= r.json()
        with open(os.path.join(seeds_path, "seeds.json"), "wb") as f:
            f.write(receive.content)
        print("Seeds are downloaded!")


    def download_preprocessed(self):
        pproc_path = self.config["preprocessed"]["dir_path"]
        if not os.path.isdir(pproc_path):
            os.makedirs(pproc_path)
            print("Created folder : ", pproc_path)

        # pproc_data
        destination = os.path.join(pproc_path, "processed.zip")
        out_file_path = pproc_path
        file_id = "1-829_LhP213j5-Xthwnj-CAxz9VC3GTH"
        self.unzip_folder(destination, out_file_path, file_id)


    def download_models(self):
        models_path = self.config["models"]["dir_path"]
        if not os.path.isdir(models_path):
            os.makedirs(models_path)
            print("Created folder : ", models_path)

        # Download the embeddings trained on the GoogleNews dataset
        g_news_path = os.path.join(models_path, self.config["models"]["google_news_subpath"])
        destination = g_news_path + ".bin.gz"
        out_file_path = g_news_path + ".bin"
        file_id = "0B7XkCwpI5KDYNlNUTTlSS21pQmM"
        self.download_and_unzip(destination, out_file_path, file_id)

        # Download the embeddings trained on the NYT dataset
        for i, file_id in zip(
            [0, 10, 100],
            [
                "14eLHQ7oo1_V6DT8h_cd69-j4e-dF7Ymg",
                "1JXzX0Egg0Hw8YpoQexc1qJG6KTK929jE",
                "1LHdwfpvPKI02kYpzTqepMKkK3xOHEyXD",
            ],
        ):
            nyt_path = os.path.join(
                models_path, self.config["models"]["nyt_subpath"][str(i)]
            )
            destination = nyt_path + ".zip"
            out_file_path = models_path
            self.unzip_folder(destination, out_file_path, file_id)

        # Download the embeddings trained on Goodreads history biography reviews
        for i, file_id in zip(
            [0, 10],
            [
                "1COecvAc3pjcIG7vpy6mGYTB28wc0gu4F",
                "1yD3Q4dfWRfQIa6VSMwqgmKD5i91KoFEL",
            ],
        ):
            history_biography_path = os.path.join(
                models_path, self.config["models"]["goodreads_hb_subpath"][str(i)]
            )
            destination = history_biography_path + ".zip"
            out_file_path = models_path
            self.unzip_folder(destination, out_file_path, file_id)

        # Download the embeddings trained on Goodreads romance reviews
        for i, file_id in zip(
            [0, 10],
            [
                "1l1W9VKjmJVUtzE6dZYfgh0RzVRszYPPU",
                "1LZOiQSWvl82qglCTZSp-nVurUmSO6qPj",
            ],
        ):
            romance_path = os.path.join(
                models_path, self.config["models"]["goodreads_r_subpath"][str(i)]
            )
            destination = romance_path + ".zip"
            out_file_path = models_path
            self.unzip_folder(destination, out_file_path, file_id)

        # Download the embeddings trained on wikitext
        for i, file_id in zip(
            [0, 10],
            [
                "1Y4_dQE_tbXun2YSFHllePv4IrVPtOTJB",
                "1KNRsNnIdtic-kch8XzE3s0ukix6gt4Dp",
            ],
        ):
            wiki_path = os.path.join(
                models_path, self.config["models"]["wiki_subpath"][str(i)]
            )
            destination = wiki_path + ".zip"
            out_file_path = models_path
            self.unzip_folder(destination, out_file_path, file_id)


    def download_all(self):
        self.download_raw()
        self.download_seeds()
        self.download_preprocessed()
        self.download_models()
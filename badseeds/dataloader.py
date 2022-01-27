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


def download_and_unzip(destination, out_file_path, file_id):

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


def unzip_folder(destination, out_file_path, file_id):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Downloads and unzips the datasets")
    parser.add_argument(
        "-c",
        "--config",
        default="./config.json",
        help="path to JSON config file outlying directory paths",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-r",
        "--raw",
        action="store_true",
        help="download raw data",
        default=False,
    )
    parser.add_argument(
        "-p",
        "--preprocessed",
        action="store_true",
        help="download preprocessed data",
        default=False,
    )
    parser.add_argument(
        "-s",
        "--seeds",
        action="store_true",
        help="download seeds",
        default=False,
    )
    parser.add_argument(
        "-m",
        "--models",
        action="store_true",
        help="download pretrained model embeddings",
        default=False,
    )
    # parse args
    args = parser.parse_args()
    # read config file
    with open(args.config, "r") as f:
        config = json.load(f)
    # ensure at least one of the flags is set
    if not (args.raw or args.preprocessed or args.seeds or args.models):
        print(
            "Please specify at least one of the following: --raw, --preprocessed, --seeds, --models"
        )
        sys.exit(1)

    if args.raw:
        raw_path = config["raw"]["dir_path"]
        # Create folder for data
        if not os.path.isdir(raw_path):
            os.makedirs(raw_path)
            print("Created folder : ", raw_path)
        # Download the WikiText dataset
        wiki_path = os.path.join(raw_path, config["raw"]["wiki_subpath"])
        if not os.path.isdir(wiki_path):
            print("Downloading the WikiText103 dataset")
            train, valid, test = torchtext.datasets.WikiText103(
                root=wiki_path, split=("train", "valid", "test")
            )
            print("Finished downloading the WikiText103 dataset")
        # Download the GoodReads History Biography dataset
        gr_hb_path = os.path.join(raw_path, config["raw"]["goodreads_hb_subpath"])
        destination = gr_hb_path + ".json.gz"
        out_file_path = gr_hb_path + ".json"
        file_id = "1lDkTzM6zpYU-HGkVAQgsw0dBzik-Zde9"
        download_and_unzip(destination, out_file_path, file_id)
        # Download the GoodReads Romance dataset
        gr_r_path = os.path.join(raw_path, config["raw"]["goodreads_r_subpath"])
        destination = gr_r_path + ".json.gz"
        out_file_path = gr_r_path + ".json"
        file_id = "1NpFsDQKBj_lrTzSASfyKbmkSykzN88wE"
        download_and_unzip(destination, out_file_path, file_id)
        # Download the NYT dataset
        nyt_path = os.path.join(raw_path, config["raw"]["nyt_subpath"])
        destination = nyt_path + ".txt.gz"
        out_file_path = nyt_path + ".txt"
        file_id = "1-2LL6wgTwDzTKfPx3RQrXi-LS6lraFYn"
        download_and_unzip(destination, out_file_path, file_id)

    if args.seeds:
        seeds_path = config["seeds"]["dir_path"]
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

    if args.preprocessed:
        pproc_path = config["preprocessed"]["dir_path"]
        if not os.path.isdir(pproc_path):
            os.makedirs(pproc_path)
            print("Created folder : ", pproc_path)

        # pproc_data
        destination = os.path.join(pproc_path, "processed.zip")
        out_file_path = pproc_path
        file_id = "1-829_LhP213j5-Xthwnj-CAxz9VC3GTH"
        unzip_folder(destination, out_file_path, file_id)

    if args.models:
        models_path = config["models"]["dir_path"]
        if not os.path.isdir(models_path):
            os.makedirs(models_path)
            print("Created folder : ", models_path)

        # Download the GoogleNews dataset
        g_news_path = os.path.join(models_path, config["models"]["google_news_subpath"])
        destination = g_news_path + ".bin.gz"
        out_file_path = g_news_path + ".bin"
        file_id = "0B7XkCwpI5KDYNlNUTTlSS21pQmM"
        download_and_unzip(destination, out_file_path, file_id)

        gr_hb_path_10 = os.path.join(
            models_path, config["models"]["goodreads_hb_subpath"]["10"]
        )
        destination = gr_hb_path_10 + ".zip"
        out_file_path = models_path
        file_id = "1yD3Q4dfWRfQIa6VSMwqgmKD5i91KoFEL"
        unzip_folder(destination, out_file_path, file_id)

        # Download the NYT dataset min frequency 10
        nyt_path_10 = os.path.join(models_path, config["models"]["nyt_subpath"]["10"])
        destination = nyt_path_10 + ".zip"
        out_file_path = models_path
        file_id = "1JXzX0Egg0Hw8YpoQexc1qJG6KTK929jE"
        unzip_folder(destination, out_file_path, file_id)

        # Download the NYT dataset min frequency 100
        nyt_path_100 = os.path.join(models_path, config["models"]["nyt_subpath"]["100"])
        destination = nyt_path_100 + ".zip"
        out_file_path = models_path
        file_id = "1LHdwfpvPKI02kYpzTqepMKkK3xOHEyXD"
        unzip_folder(destination, out_file_path, file_id)

        # Download the romance dataset
        romance_path_10 = os.path.join(
            models_path, config["models"]["goodreads_r_subpath"]["10"]
        )
        destination = romance_path_10 + ".zip"
        out_file_path = models_path
        file_id = "1LZOiQSWvl82qglCTZSp-nVurUmSO6qPj"
        unzip_folder(destination, out_file_path, file_id)

        # Download the wiki_train_tokens_min10 dataset
        wiki_path_10 = os.path.join(models_path, config["models"]["wiki_subpath"]["10"])
        destination = wiki_path_10 + ".zip"
        out_file_path = models_path
        file_id = "1KNRsNnIdtic-kch8XzE3s0ukix6gt4Dp"
        unzip_folder(destination, out_file_path, file_id)

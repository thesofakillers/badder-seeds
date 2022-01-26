import os
import torchtext
import gzip
import shutil
import gdown
from zipfile import ZipFile
import requests
import getopt
import sys


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

    # insure commandline arguments
    if len(sys.argv) <= 1:
        print(
            "Please specify with one of the following arguments to download: \n --raw \n --cleaned \n --seeds \n --pretrained \n"
        )
        exit(1)

    # Remove 1st argument from the
    # list of command line arguments
    argumentList = sys.argv[1:]

    # Options
    options = "rcps:"

    # Long options
    long_options = ["raw", "cleaned", "seeds", "pretrained"]

    try:
        # Parsing argument
        arguments, values = getopt.getopt(argumentList, options, long_options)

        for currentArgument, currentValue in arguments:

            if currentArgument in ("-r", "--raw"):

                # Create folder for data
                if not os.path.isdir("../data/raw"):
                    os.makedirs("../data/raw")
                    print("Created folder : ", "../data/raw")

                # Download the WikiText dataset
                if not os.path.isdir("../data/WikiText103"):
                    print("Downloading the WikiText103 dataset")
                    train, valid, test = torchtext.datasets.WikiText103(
                        root="../data", split=("train", "valid", "test")
                    )
                    print("Finished downloading the WikiText103 dataset")

                # Download the GoodReads History Biography dataset
                destination = "../data/raw/goodreads_reviews_history_biography.json.gz"
                out_file_path = "../data/raw/goodreads_reviews_history_biography.json"
                file_id = "1lDkTzM6zpYU-HGkVAQgsw0dBzik-Zde9"
                download_and_unzip(destination, out_file_path, file_id)

                # Download the GoodReads Romance dataset
                destination = "../data/raw/goodreads_reviews_romance.json.gz"
                out_file_path = "../data/raw/goodreads_reviews_romance.json"
                file_id = "1NpFsDQKBj_lrTzSASfyKbmkSykzN88wE"
                download_and_unzip(destination, out_file_path, file_id)

                # Download the GoogleNews dataset
                destination = "../data/raw/GoogleNews-vectors-negative300.bin.gz"
                out_file_path = "../data/raw/GoogleNews-vectors-negative300.bin"
                file_id = "0B7XkCwpI5KDYNlNUTTlSS21pQmM"
                download_and_unzip(destination, out_file_path, file_id)

                # Download the NYT dataset
                destination = "../data/raw/nytimes_news_articles.txt.gz"
                out_file_path = "../data/raw/nytimes_news_articles.txt"
                file_id = "1-2LL6wgTwDzTKfPx3RQrXi-LS6lraFYn"
                download_and_unzip(destination, out_file_path, file_id)

            if currentArgument in ("-s", "--seeds"):
                # Create folder for seed data
                if not os.path.isdir("../data/seeds"):
                    os.makedirs("../data/seeds")
                    print("Created folder : ", "../data/seeds")

                # download seeds
                receive = requests.get(
                    "https://raw.githubusercontent.com/maria-antoniak/bad-seeds/main/gathered_seeds.json"
                )
                # r_dictionary= r.json()
                with open(r"../data/seeds/seeds.json", "wb") as f:
                    f.write(receive.content)
                print("Seeds are downloaded!")

            if currentArgument in ("-c", "--cleaned"):
                # download cleaned data
                if not os.path.isdir("../data/preprocessed_data"):
                    os.makedirs("../data/preprocessed_data")
                    print("Created folder : ", "../data/preprocessed_data")

                # cleaned data
                destination = "../data/preprocessed_data/processed.zip"
                out_file_path = "../data/preprocessed_data/"
                file_id = "1-829_LhP213j5-Xthwnj-CAxz9VC3GTH"
                unzip_folder(destination, out_file_path, file_id)

            if currentArgument in ("-p", "--pretrained"):
                if not os.path.isdir("../data/models"):
                    # download pretrained embeddings of unigram
                    os.makedirs("../data/models")
                    print("Created folder : ", "../data/models")

                destination = "../data/models/history_biography_min10.zip"
                out_file_path = "../data/models/"
                file_id = "1yD3Q4dfWRfQIa6VSMwqgmKD5i91KoFEL"
                unzip_folder(destination, out_file_path, file_id)

                # Download the NYT dataset min frequency 10
                destination = "../data/models/nytimes_news_articles_min10.zip"
                out_file_path = "../data/models/"
                file_id = "1JXzX0Egg0Hw8YpoQexc1qJG6KTK929jE"
                unzip_folder(destination, out_file_path, file_id)

                # Download the NYT dataset min frequency 100
                destination = "../data/models/nytimes_news_articles_min100.zip"
                out_file_path = "../data/models/"
                file_id = "1LHdwfpvPKI02kYpzTqepMKkK3xOHEyXD"
                unzip_folder(destination, out_file_path, file_id)

                # Download the romance dataset
                destination = "../data/models/romance_min10.zip"
                out_file_path = "../data/models/"
                file_id = "1LZOiQSWvl82qglCTZSp-nVurUmSO6qPj"
                unzip_folder(destination, out_file_path, file_id)

                # Download the wiki_train_tokens_min10 dataset
                destination = "../data/models/wiki.train.tokens_min10.zip"
                out_file_path = "../data/models/"
                file_id = "1KNRsNnIdtic-kch8XzE3s0ukix6gt4Dp"
                unzip_folder(destination, out_file_path, file_id)

    except getopt.error as err:
        # output error, and return with an error code
        print(str(err))

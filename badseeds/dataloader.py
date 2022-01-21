import os
import torchtext
import gzip
import shutil
import gdown
from zipfile import ZipFile
import requests


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


if __name__ == "__main__":
    # Create folder for data
    if not os.path.isdir("../data"):
        os.makedirs("../data")
        print("Created folder : ", "../data")

    # Download the WikiText dataset
    if not os.path.isdir("../data/WikiText103"):
        print("Downloading the WikiText103 dataset")
        train, valid, test = torchtext.datasets.WikiText103(
            root="../data", split=("train", "valid", "test")
        )
        print("Finished downloading the WikiText103 dataset")

    # Download the GoodReads History Biography dataset
    destination = "../data/goodreads_reviews_history_biography.json.gz"
    out_file_path = "../data/goodreads_reviews_history_biography.json"
    file_id = "1lDkTzM6zpYU-HGkVAQgsw0dBzik-Zde9"
    download_and_unzip(destination, out_file_path, file_id)

    # Download the GoodReads Romance dataset
    destination = "../data/goodreads_reviews_romance.json.gz"
    out_file_path = "../data/goodreads_reviews_romance.json"
    file_id = "1NpFsDQKBj_lrTzSASfyKbmkSykzN88wE"
    download_and_unzip(destination, out_file_path, file_id)

    # Download the GoogleNews dataset
    destination = "../data/GoogleNews-vectors-negative300.bin.gz"
    out_file_path = "../data/GoogleNews-vectors-negative300.bin"
    file_id = "0B7XkCwpI5KDYNlNUTTlSS21pQmM"
    download_and_unzip(destination, out_file_path, file_id)

    # Download the NYT dataset
    destination = "../data/nytimes_news_articles.txt.gz"
    out_file_path = "../data/nytimes_news_articles.txt"
    file_id = "1ITZ6FZq4_C2hs7k540ZYiReNTlWGt4nz"
    download_and_unzip(destination, out_file_path, file_id)

    # Download Gnews small dataset
    destination = "../data/w2v_gnews_small.zip"
    out_file_path = "../data/w2v_gnews_small.txt"
    file_id = "1NH6jcrg8SXbnhpIXRIXF_-KUE7wGxGaG"
    download_and_unzip(destination, out_file_path, file_id)

    # Create folder for seed data
    if not os.path.isdir("../data/seeds"):
        os.makedirs("../data/seeds")
        print("Created folder : ", "../data/seeds")

    # download seeds
    recieve = requests.get(
        "https://raw.githubusercontent.com/maria-antoniak/bad-seeds/main/gathered_seeds.json"
    )
    # r_dictionary= r.json()
    with open(r"../data/seeds/seeds.json", "wb") as f:
        f.write(recieve.content)

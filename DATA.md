# Downloadable Datasets and Models

The work in this repository use of a number of raw and processed datasets and
models. To run many of the notebooks, these datasets need to be downloaded. We
provide the python script [dataloader.py](badseeds/dataloader.py) for
automatically downloading the necessary raw data. The script takes arguments
such that the user can specify which of the datasets to download.

Users can alternatively manually download the datasets
themselves from the following links, organized by dataset kind.

Note that throughout the work, the data directory structure is assumed to be in
the format specified in [config.json](config.json), which is what
[dataloader.py](badseeds/dataloader.py) follows by default. If users wish to
reorganize their data paths, simply ensure that [config.json](config.json) is
edited accordingly to reflect these changes.

## Raw Datasets

- [New York Times](https://drive.google.com/file/d/1-2LL6wgTwDzTKfPx3RQrXi-LS6lraFYn/view?usp=sharing)
- [WikiText 103](https://torchtext.readthedocs.io/en/latest/datasets.html#wikitext103)
- [Goodreads Romance
  reviews](https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/home#h.p_T_Y3TfiAzl14)
- [Goodreads History and Biography
  reviews](https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/home#h.p_SlyLpu83zdEn)

## Preprocessed Datasets

The majority of the notebooks make use of a preprocessed version of the
datasets above. We provide a link to the preprocessed version of these datasets
to allow users to skip downloading the raw data files and skip preprocessing,
which is a time consuming operation. The preprocessed datasets can be found
[here](https://drive.google.com/file/d/1-829_LhP213j5-Xthwnj-CAxz9VC3GTH/view?usp=sharing)

## Seeds

The work of course makes use of the seedbank provided by Antoniak and Mimno,
available for download from their
[repository](https://raw.githubusercontent.com/maria-antoniak/bad-seeds/main/gathered_seeds.json)

## Models

For users who wish to skip model training, and simply reproduce the plots, we
provide the pretrained embeddings for download
[here](https://drive.google.com/drive/folders/13-_uejiF1QH_mOgp_2djfU__H7VtZScO?usp=sharing)

We also direct users
[here](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/view?usp=sharing&resourcekey=0-wjGZdNAUop6WykTtMip30g)
for downloading the pretrained GoogleNews embedding from Mikolov et al. 2013

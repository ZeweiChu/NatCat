# Data

## NatCat

This repo provides the NatCat dataset. NatCat are naturally annotated category-text pairs for training text classifiers. 

NatCat can be downloaded from [here](https://drive.google.com/file/d/1ej45NfTy1hhNFJGqPbAyrrLQS6b3uMIf/view?usp=sharing)

NatCat is constructed from three different data domains. 
- Wikipedia
- Stack Exchange
- Reddit

Each directory contains the data from the corresponding domain. The files are named as train.tsv???.data 
Each file is tab separated data. The first field is the positive/correct category, the second to the eighth fields are negative/wrong categories. The nineth/last field is the text to categorize. 

We provide the sampled data we used in our paper under [sample-data](/data/sample-data).


## CatEval

[CatEval](/data/cateval) contains the 11 tasks we use to evaluate NatCat trained text classifiers. 

Under each task directory, the file named ```classes.txt.acl``` list the category names we used to run the experiments. 

## WikiCat

We provide another full version of NatCat constructed from Wikipeda, namely, WikiCat. It can be downloaded from [here](https://drive.google.com/file/d/1N8WlbpG0p90GMQup7Bq3Kp8Y4NapBw3T/view?usp=sharing). 

WikiCat is constructed from Wikipedia. Each Wikipedia page is annotated by categories (can be found at the bottom of each Wikipedia page) and their immediate parent categories.

WikiCat can be used to train topical text classification models.

### Files
- wikipedia-documents: contains all Wikpedia documents. Each file is named by a digital ID and contains a single Wikipedia document.
- {train,dev}.tsv are tab separated files containing Wikipedia IDs and their corresponding categories. Each row starts with a Wikipedia ID, and followed by their annotated categories separated by tabs.


Zewei Chu
9/29/2020

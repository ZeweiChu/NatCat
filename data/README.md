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

Zewei Chu
9/29/2020

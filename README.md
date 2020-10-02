# NatCat

This repo provides the NatCat dataset from [Natcat: Weakly Supervised Text Classification with Naturally Annotated Datasets](https://arxiv.org/abs/2009.14335). 

## Data

### NatCat
NatCat can be downloaded from [here](https://drive.google.com/file/d/1ej45NfTy1hhNFJGqPbAyrrLQS6b3uMIf/view?usp=sharing)

NatCat are naturally annotated category-text pairs for training text classifiers. 

NatCat is constructed from three different data domains. 
- Wikipedia
- Stack Exchange
- Reddit

Each directory contains the data from the corresponding domain. The files are named as ```train.tsv???.data```
Each data file is tab separated data. The first field is the positive/correct category, the second to the eighth fields are negative/wrong categories. The nineth/last field is the text to categorize. 

### CatEval

[CatEval](/data/cateval) contains the 11 tasks we use to evaluate NatCat trained text classifiers. 

Under each task directory, the file named ```classes.txt.acl``` list the category names we used to run the experiments. 

### WikiCat

We provide another full version of NatCat constructed from Wikipeda, namely, WikiCat. It can be downloaded from [here](https://drive.google.com/file/d/1N8WlbpG0p90GMQup7Bq3Kp8Y4NapBw3T/view?usp=sharing). 

WikiCat is constructed from Wikipedia. Each Wikipedia page is annotated by categories (can be found at the bottom of each Wikipedia page) and their immediate parent categories.

WikiCat can be used to train topical text classification models.

#### Files
- wikipedia-documents: contains all Wikpedia documents. Each file is named by a digital ID and contains a single Wikipedia document.
- {train,dev}.tsv are tab separated files containing Wikipedia IDs and their corresponding categories. Each row starts with a Wikipedia ID, and followed by their annotated categories separated by tabs.


## Code

```bash
python code/run_natcat.py \
    --model_type roberta \
    --model_name_or_path roberta-large \
    --task_name natcat \
    --seed 1 \
    --do_train \
    --do_lower_case \
    --data_dir data/sample-data \
    --max_seq_length 128 \
    --per_gpu_train_batch_size=16   \
    --learning_rate 2e-5 \
    --num_train_epochs 1 \
    --save_total_limit 3 \
    --output_dir saved_checkpoints/roberta-large \
    --warmup_steps 15000
```

```bash
python code/run_eval.py \
    --model_type roberta \
    --model_name_or_path saved_checkpoints/roberta-large \
    --task_name eval \
    --do_eval \
    --do_lower_case \
    --eval_data_file data/cateval/agnews/test.csv \
    --max_seq_length 128 \
    --class_file_name=data/cateval/agnews/classes.txt.acl \
    --pred_output_file=saved_checkpoints/roberta-large/agnews.preds.txt \
    --output_dir saved_checkpoints/roberta-large \
    --per_gpu_eval_batch_size=64 
```


Dependencies
- transformers 3.1.0
- torch 1.4.0


## Citation

Zewei Chu
9/29/2020

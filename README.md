# NatCat

This repo provides the NatCat dataset from [Natcat: Weakly Supervised Text Classification with Naturally Annotated Datasets](). 

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

## Citation

Zewei Chu
9/29/2020

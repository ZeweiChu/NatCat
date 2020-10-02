import os
from transformers.data.processors.utils import DataProcessor, InputExample
from tqdm import tqdm
import random
import code

class NatCatProcessor(DataProcessor):
    """Processor for the NatCat data set."""  
    def __init__(self):
        super(NatCatProcessor, self).__init__()

    def get_examples(self, filepath):
        """See base class."""
        """
            filepath: the file of article-category pairs 
        """
        examples = []
        i = 0
        with open(filepath) as fin:
            lines = fin.read().strip().split("\n")
            for line in tqdm(lines):
                line = line.strip().split("\t")

                pos_cats = line[:1]
                neg_cats = line[len(pos_cats):-1]
                article = line[-1]
                for pos_cat in pos_cats:
                    examples.append(InputExample(guid=i, text_a=pos_cat, text_b=article, label='1'))
                    i += 1
                for neg_cat in neg_cats:
                    examples.append(InputExample(guid=i, text_a=neg_cat, text_b=article, label='0'))
                    i += 1


        return examples 

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

class EvalProcessor:
    def __init__(self, cat_file_path):
        super(EvalProcessor, self).__init__()
        self.cats = []
        with open(cat_file_path) as fin:
            for line in fin.read().strip().split("\n"):
                self.cats.append(line.strip())

    def get_examples(self, filepath):
        """See base class."""
        """
            filepath: the file of the evaluation dataset 
        """
        examples = []
        labels = []
        i = 0
        with open(filepath) as fin:
            lines = fin.read().strip().split("\n")
            for line in tqdm(lines):
                line = line.strip().split(",", 1)
                if line[0].startswith("'") or line[0].startswith('"'):
                    line[0] = line[0][1:-1]
                label = int(line[0]) - 1
                text = " ".join(line[1][1:-1].split()[:128])
                if text.strip() == "":
                    text = "N/A"
                for cat in self.cats:
                    i += 1
                    if cat == self.cats[label]:
                        examples.append(InputExample(guid=i, text_a=cat, text_b=text, label=1))
                    else:
                        examples.append(InputExample(guid=i, text_a=cat, text_b=text, label=0))

        return examples

    def get_labels(self):
        return [0, 1]

class MultiProcessor:
    def __init__(self, cat_file_path):
        super(MultiProcessor, self).__init__()
        self.cats = []
        with open(cat_file_path) as fin:
            for line in fin:
                self.cats.append(line.strip())

    def get_examples(self, text_filepath, label_filepath):
        """ See base class."""
        """
            filepath: the file of the evaluation dataset 
        """
        examples = []
        labels = []
        i = 0
        with open(text_filepath) as fin:
            text_lines = fin.read().strip().split("\n")
        with open(label_filepath) as fin:
            label_lines = fin.read().strip().split("\n")

        for text, labels in tqdm(zip(text_lines, label_lines)):
            text = " ".join(text.strip().split()[:128])
            labels = set(labels.strip().split("\t"))
            for cat in self.cats:
                if cat in labels:
                    label = 1
                else:
                    label = 0
                examples.append(InputExample(guid=i, text_a=cat, text_b=text, label=label))

        return examples

    def get_labels(self):
        return [0, 1]


processors = {
    "natcat": NatCatProcessor,
    "eval": EvalProcessor,
    "nyt": MultiProcessor,
    "comment": MultiProcessor,
    "situation": MultiProcessor,
}

output_modes = {
    "natcat": "classification",
    "eval": "classification",
    "nyt": "classification",
    "comment": "classification",
    "situation": "classification",
}

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def compute_metrics(task_name, preds, labels):
    if task_name in ["natcat"]:
        return {"acc": simple_accuracy(preds, labels)}
    if task_name in ["eval"]:
        return {"acc": simple_accuracy(preds, labels)}
    if task_name in ["nyt"]:
        return {"acc": simple_accuracy(preds, labels)}
    if task_name in ["comment"]:
        return {"acc": simple_accuracy(preds, labels)}
    if task_name in ["situation"]:
        return {"acc": simple_accuracy(preds, labels)}

class DataFiles:
    def __init__(self, directory):
        self.all_files = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and f.endswith(".data")]
        self.todo_files = self.all_files

    def next(self):
        if len(self.todo_files) == 0:
            return None
        return self.todo_files.pop()

    def save(self, file_path):
        with open(file_path, "w") as fout:
            for f in self.todo_files:
                fout.write(f + "\n")

    def load(self, file_path):
        self.todo_files = []
        with open(file_path) as fin:
            for f in fin:
                self.todo_files.append(f.strip())



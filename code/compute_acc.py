import numpy as np
import sys
from sklearn.metrics import f1_score

if len(sys.argv) != 3:
    print("Usage: python compute_acc.py preds_file test_file")
    exit(-1)

preds_file = sys.argv[1]
test_file = sys.argv[2]

with open(test_file) as fin:
    labels = []
    for line in fin:
        label, _ = line.split(",", 1)
        if label.startswith("'") or label.startswith('"'):
            label = label[1:-1]
        labels.append(int(label) - 1)

    labels = np.array(labels)

preds = np.loadtxt(preds_file)
preds = preds.argmax(1) 

acc = np.sum(preds == labels) / len(labels)

print("accuracy of {}: {}".format(preds_file, acc))

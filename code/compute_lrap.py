import numpy as np
import sys
from sklearn.metrics import label_ranking_average_precision_score


if len(sys.argv) != 4:
    print("Usage: python compute_lrap.py preds_file test_file all_class_file")
    exit(-1)

preds_file = sys.argv[1]
test_file = sys.argv[2]
all_class_file = sys.argv[3]

class2index = {}
with open(all_class_file) as fin:
    for line in fin:
        class2index[line.strip()] = len(class2index)
        
y_score = np.loadtxt(preds_file)
y_true = np.zeros(y_score.shape)

i = 0
with open(test_file) as fin:
    for line in fin:
        for c in line.strip().split("\t"):
            y_true[i, class2index[c]] = 1
        i += 1


lrap = label_ranking_average_precision_score(y_true, y_score)

print("lrap of {}: {}".format(preds_file, lrap))

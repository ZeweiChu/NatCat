from random import shuffle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--lines', type=int, default=100, help="number of categories to use for the classification task")
parser.add_argument('--num_train_samples', type=int, default=1000000, help="the number of train examples")
parser.add_argument('--num_dev_samples', type=int, default=10000, help="the number of dev examples")
parser.add_argument('--num_test_samples', type=int, default=10000, help="the number of test examples")
args = parser.parse_args()

classes_to_remove = set()
with open("../tools/categories_to_remove.txt", "r") as fin:
	for line in fin:
		line = line.strip()
		classes_to_remove.add(line)
	
all_classes = set()
num_lines = args.lines
i = 0
with open("../class-counts.txt", "r") as fin:
	for line in fin:
		line = line.strip().split(" ", 1)[1]
		if line in classes_to_remove:
			continue
		all_classes.add(line)
		i += 1
		if i >= num_lines:
			break

# now we have a list of classes to classify into

def preprocess_category(cat):
	cat = cat.strip().split("/")[-1].lower()
	return cat

docs = []
classes = []

with open("../docs.txt", "r") as fin:
	for line in fin:
		docs.append(line.strip())

with open("../classes.txt", "r") as fin:
	for line in fin:
		classes.append(line.strip().split("\t"))

doc_class_pairs = list(zip(docs, classes))

shuffle(doc_class_pairs)

i = 0
num_train_samples = args.num_train_samples
num_dev_samples = args.num_dev_samples
num_test_samples = args.num_test_samples
all_train_pairs = []
all_dev_pairs = []
all_test_pairs = []

for doc, classes in doc_class_pairs:
	new_classes = set()
	for c in classes:
		c = preprocess_category(c)
		if c in all_classes:
			new_classes.add(c)
	if len(new_classes) > 0:
		if i < num_train_samples:
			all_train_pairs.append([doc, new_classes])
		elif i < num_train_samples + num_dev_samples:
			all_dev_pairs.append([doc, new_classes])
		elif i < num_train_samples + num_dev_samples + num_test_samples:
			all_test_pairs.append([doc, new_classes])
		else:
			break
		i += 1

with open("train.doc.txt", "w") as fdoc, open("train.class.txt", "w") as fclass:
	for doc, classes in all_train_pairs:
		fdoc.write(doc + "\n")
		fclass.write("\t".join(classes) + "\n")

with open("dev.doc.txt", "w") as fdoc, open("dev.class.txt", "w") as fclass:
	for doc, classes in all_dev_pairs:
		fdoc.write(doc + "\n")
		fclass.write("\t".join(classes) + "\n")

with open("test.doc.txt", "w") as fdoc, open("test.class.txt", "w") as fclass:
	for doc, classes in all_test_pairs:
		fdoc.write(doc + "\n")
		fclass.write("\t".join(classes) + "\n")
		
with open("all_classes.txt", "w") as fout:
	for c in all_classes:
		fout.write(c + "\n")

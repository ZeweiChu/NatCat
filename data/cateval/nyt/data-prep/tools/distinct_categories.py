all_classes = set()
with open("../classes.txt", "r") as fin:
	for line in fin:
		line = line.strip().split("\t")
		for c in line:
			all_classes.add(c)

with open("../distinct_classes.txt", "w") as fout:
	for c in all_classes:
		fout.write(c + "\n")

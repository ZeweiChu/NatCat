max_length = 0
with open("../docs.txt", "r") as fin:
  for line in fin:
    length = len(line.split())
    if length > max_length:
      max_length = length

print(max_length)

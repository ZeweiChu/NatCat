cat distinct_classes.txt | rev | cut -d"/" -f1 | rev | uniq -c | sort -g

cat distinct_classes.txt | rev | cut -d"/" -f1 | rev | tr '[:upper:]' '[:lower:]' | sort | uniq | wc -l


# count the number of documents belonging to each distinct category
sed -e 'y/\t/\n/' classes.txt | rev | cut -d"/" -f1 | rev | tr '[:upper:]' '[:lower:]' | sort | uniq -c | sort -g -r  > class-counts.txt 




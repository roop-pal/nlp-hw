# Part 3

# Use modified file generated by
# python count_freqs.py modified_ner_train.dat > modified_ner.counts

# Part 1
from q4p12 import D
from math import log

word_tag = D()
grams = {}
for line in open('modified_ner.counts', 'r'):
    line = line.strip('\n').split(' ')
    if line[1] == 'WORDTAG':
        word_tag[line[3]][line[2]] = int(line[0])
    elif line[1] == '1-GRAM':
        grams[line[2]] = int(line[0])

def e(x, y):
    if y not in word_tag[x]:
        return 0
    return word_tag[x][y] / grams[y]

word_counts = {}
train = open('ner_train.dat', 'r')
for line in train:
    line = line.strip('\n').split(' ')
    if len(line[0]) > 0:
        if line[0] not in word_counts:
            word_counts[line[0]] = 0
        word_counts[line[0]] += 1

def tag(x):
    original = x
    if x not in word_tag or word_counts[x] < 5:
        x = '_RARE_'
    max_y = ''
    max_e = 0
    for y in grams.keys():
        curr_e = e(x, y)
        if curr_e > max_e:
            max_e = curr_e
            max_y = y
    return original, max_y, str(log(max_e))

with open('prediction_file.txt', 'w') as outfile:
    for line in open("ner_dev.dat", 'r'):
        line = line.strip("\n")
        if line:
            outfile.write(' '.join(tag(line)))
        outfile.write('\n')
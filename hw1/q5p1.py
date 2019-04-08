from q4p12 import D
from math import log

# Part 1
word_tag = D()
grams = D()
K = set([])
for line in open('modified_ner.counts', 'r'):
    line = line.strip('\n').split(' ')
    if line[1] == 'WORDTAG':
        word_tag[line[3]][line[2]] = int(line[0])
    else: #-GRAM
        K.add(line[2])
        grams[''.join(line[2:])] = int(line[0])


def e(x, y):
    if y not in word_tag[x]:
        return 0
    return word_tag[x][y] / grams[y]


def q(y3, y1, y2):
    if y1 + y2 + y3 not in grams:
        return 0
    return grams[y1 + y2 + y3] / grams[y1 + y2]


with open('5_1.txt', 'w') as outfile:
    for line in open('trigrams.txt', 'r'):
        line = line.strip('\n').split(' ')
        outfile.write(line[0] + ' ' + line[1] + ' ' + line[2] + ' ')
        outfile.write(str(log(q(line[2], line[0], line[1]))) + '\n')
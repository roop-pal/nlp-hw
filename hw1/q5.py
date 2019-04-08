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
    return word_tag[x][y] / grams[[y]]


def q(y3, y1, y2):
    if y1 + y2 + y3 not in grams:
        return 0
    return grams[y1 + y2 + y3] / grams[y1 + y2]

# print(q('O','*','*'))

# Part 2

def viterbi(x):
    pi = {}
    bp = {}
    pi[(0,'*','*')] = 0
    K0 = ['*']
    K1 = ['*']
    for k in range(1, len(x) + 1):
        if k == 2:
            K1 = K
        if k == 3:
            K0 = K
        for u in K1:
            for v in K:
                for w in K0:
                    if (k,u,v) not in pi:
                        pi[(k,u,v)] = log(q(v, w, u)) + log(e(x[k-1], v)) +\
                                       pi[(k-1,w,u)]



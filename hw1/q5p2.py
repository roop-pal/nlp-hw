from math import log
from q5p1 import q,e

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
                pi[(k, u, v)] = 0
                bp[(k, u, v)] = '*'
                for w in K0:
                    # if (k,u,v) not in pi:
                    if e(x[k-1], v) == 0 or q(v, w, u) == 0:
                        p = float('-inf')
                    else:
                        p = log(q(v, w, u)) + log(e(x[k-1], v)) +\
                                   pi[(k-1,w,u)]
                    if p > pi[(k,u,v)]:
                        pi[(k,u,v)] = p
                        bp[(k,u,v)] = w
    # print(pi)
    print(bp)

viterbi(['*','*','prime','minister'])
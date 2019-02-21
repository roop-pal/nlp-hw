from math import log
from q5p1 import q,e
from q4p12 import D

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
    for k in range(1, len(x)):
        if k == 2:
            K1 = K
        if k == 3:
            K0 = K
        for u in K1:
            for v in K:
                pi[(k, u, v)] = float('-inf')
                bp[(k, u, v)] = '*'
                for w in K0:
                    word = x[k - 1]
                    if word not in word_tag:
                        word = '_RARE_'
                    if e(word, v) == 0 or q(v, w, u) == 0:
                        p = float('-inf')
                    else:
                        p = log(q(v, w, u)) + log(e(word, v)) + pi[(k - 1, w, u)]
                    if p > pi[(k,u,v)]:
                        pi[(k,u,v)] = p
                        bp[(k,u,v)] = w

    p_max = float('-inf')
    u_max, v_max = None, None
    for u in K:
        for v in K:
            if len(x) == 2:
                p = pi[(len(x) - 1, '*', v)]
            else:
                p = pi[(len(x) - 1, u, v)]
            if p > p_max:
                p_max = p
                u_max = u
                v_max = v

    labels = [u_max, v_max]
    for k in range(len(x) - 3, 0, -1):
        yk = bp[(k + 2, labels[0], labels[1])]
        labels = [yk] + labels

    pis = []
    for k in range(len(x) - 1, 1, -1):
        pis = [pi[(k, labels[k - 2], labels[k - 1])]] + pis
    pis = [pi[(1, '*', labels[0])]] + pis

    return labels, pis

with open('5_2.txt', 'w') as outfile:
    x = []
    for line in open('ner_dev.dat', 'r'):
        line = line.strip('\n')
        x.append(line)
        if not line:
            labels, pis = viterbi(x)
            for i in range(len(x) - 1):
                outfile.write(x[i] + ' ' + labels[i] + ' ' + str(pis[i]) + '\n')
            outfile.write('\n')
            x = []

# PERFORMANCE
# Found 4661 NEs. Expected 5931 NEs; Correct: 3657.
#
#          precision      recall          F1-Score
# Total:   0.784596       0.616591        0.690521
# PER:     0.744311       0.605005        0.667467
# ORG:     0.659729       0.473842        0.551544
# LOC:     0.887883       0.695202        0.779817
# MISC:    0.825974       0.690554        0.752218

# The performance is not terrible, with higher precision than recall.
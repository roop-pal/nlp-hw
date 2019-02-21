from math import log
from q4p12 import D
from q6p1 import classify

word_tag = D()
grams = D()
K = set([])
for line in open('q6_ner.counts', 'r'):
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
                        word = classify(word)
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

with open('6.txt', 'w') as outfile:
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
# Found 5556 NEs. Expected 5931 NEs; Correct: 4273.
#
#          precision      recall          F1-Score
# Total:   0.769078       0.720452        0.743971
# PER:     0.813967       0.773667        0.793305
# ORG:     0.584724       0.680867        0.629144
# LOC:     0.869128       0.706107        0.779182
# MISC:    0.847569       0.700326        0.766944

# Though the precision decreased, the recall greatly increased, giving the best f1-scores.
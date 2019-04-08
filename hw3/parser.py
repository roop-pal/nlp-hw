import sys, os, json


# dict with useful properties i.e. supporting d[0][1] += 1 operation (also works for lists
class D(dict):
    def __add__(self, other):
        if self == {}:
            return other
        # return self + other

    def __getitem__(self, item):
        try:
            return dict.__getitem__(self, item)
        except KeyError:
            value = self[item] = type(self)()
            return value


def replace(t, wc):
    if len(t) == 3:
        replace(t[1], wc)
        replace(t[2], wc)
    else:
        if wc[t[1]] < 5:
            t[1] = '_RARE_'
    return t


def q4():
    word_counts = D()
    for line in open('cfg.counts', 'r'):
        line = line.strip('\n').split(' ')
        if line[1] == 'UNARYRULE':
            word_counts[line[3]] += int(line[0])

    with open(sys.argv[3], 'w') as outfile:
        for line in open(sys.argv[2], 'r'):
            outfile.write(json.dumps(replace(json.loads(line), word_counts)) + '\n')


def tree(i, j, S, s, bp):
    t = [S]
    if i == j:
        t.append(s[i])
    else:
        l, r = bp[(i, j, S)]
        t += [tree(*l, s, bp)] + [tree(*r, s, bp)]
    return t


def cky(sentence, N, R1, R2, q, wc):
    n = len(sentence)
    pi = {}
    for i in range(n):
        for X in N:
            if (X, sentence[i]) in R1:
                pi[(i, i, X)] = q[(X, sentence[i])]
            elif (X, '_RARE_') in R1 and (sentence[i] not in wc or wc[sentence[i]] < 5):
                pi[(i, i, X)] = q[(X, '_RARE_')]
            else:
                pi[(i, i, X)] = 0
    bp = {}
    for l in range(1, n):
        for i in range(n - l):
            j = i + l
            for X in N:
                pi[(i, j, X)] = 0
                for Y, Z in R2[X]:
                    for s in range(i, j):
                        p = q[(X, Y, Z)] * pi[(i, s, Y)] * pi[(s + 1, j, Z)]
                        if p > pi[(i, j, X)]:
                            pi[(i, j, X)] = p
                            bp[(i, j, X)] = ((i, s, Y), (s + 1, j, Z))
    S = 'S'
    if pi[(0, n - 1, S)] == 0:
        p = float('-inf')
        for X in N:
            if p < pi[0, n - 1, X]:
                S = X
                p = pi[0, n - 1, X]
    return tree(i, j, S, sentence, bp)


def q5():
    gram_counts = D()
    word_counts = D()
    N, R1, R2 = [], [], D()
    for line in open('cfg.RARE.counts', 'r'):
        line = line.strip('\n').split(' ')
        if line[1] == 'UNARYRULE':
            R1.append((line[2], line[3]))
            gram_counts[(line[2], line[3])] += int(line[0])
            word_counts[line[3]] += int(line[0])
        elif line[1] == 'NONTERMINAL':
            N.append(line[2])
            gram_counts[(line[2],)] += int(line[0])
        elif line[1] == 'BINARYRULE':
            R2[line[2]] += [(line[3], line[4])]
            gram_counts[(line[2], line[3], line[4])] += int(line[0])

    q = {}
    for key in gram_counts:
        if len(key) > 1:
            q[key] = gram_counts[key] / gram_counts[(key[0],)]

    a = 0
    with open(sys.argv[4], 'w') as outfile:
        for line in open(sys.argv[3], 'r'):
            a += 1
            s = line.strip('\n').split(' ')
            outfile.write(json.dumps(cky(s, N, R1, R2, q, word_counts)) + '\n')


if __name__ == '__main__':
    assert (len(sys.argv) >= 2)
    if sys.argv[1] == 'q4':
        os.system('python count_cfg_freq3.py ' + sys.argv[2] + ' > cfg.counts')
        q4()
    elif sys.argv[1] == 'q5':
        os.system('python count_cfg_freq3.py ' + sys.argv[2] + ' > cfg.RARE.counts')
        q5()
    else:
        os.system('python count_cfg_freq3.py ' + sys.argv[2] + ' > cfg.RARE.counts')
        q5()
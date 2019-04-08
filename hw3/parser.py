import sys, os, json
from IPython import embed as e


# dict with useful properties i.e. supporting d[0][1] += 1 operation
class D(dict):
    def __add__(self, other):
        if self == {}:
            return other
        return self + other

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
        assert len(t) == 2
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


if __name__ == '__main__':
    assert (len(sys.argv) >= 2)
    if sys.argv[1] == 'q4':
        q4()
    elif sys.argv[1] == 'q5':
        print(sys.argv)
    else:
        print(sys.argv)
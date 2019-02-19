# https://stackoverflow.com/questions/651794/whats-the-best-way-to-initialize-a-dict-of-dicts-in-python
class D(dict):
    """Implementation of perl's autovivification feature."""
    def __getitem__(self, item):
        try:
            return dict.__getitem__(self, item)
        except KeyError:
            value = self[item] = type(self)()
            return value

# Part 1
word_tag = D()
grams = {}
for line in open('ner.counts', 'r'):
    line = line.strip('\n').split(' ')
    if line[1] == 'WORDTAG':
        word_tag[line[3]][line[2]] = int(line[0])
    elif line[1] == '1-GRAM':
        grams[line[2]] = int(line[0])

def e(x, y):
    if y not in word_tag[x]:
        return 0
    return word_tag[x][y] / grams[y]

# print(e('prime','O'))

# Part 2
word_counts = {}
train = open('ner_train.dat', 'r')
for line in train:
    line = line.strip('\n').split(' ')
    if len(line[0]) > 0:
        if line[0] not in word_counts:
            word_counts[line[0]] = 0
        word_counts[line[0]] += 1
train.close()

train = open('ner_train.dat', 'r')
with open('modified_ner_train.dat', 'w') as outfile:
    for line in train:
        line = line.strip('\n').split(' ')
        if len(line[0]) > 0:
            if word_counts[line[0]] < 5:
                line[0] = '_RARE_ '
        outfile.write(' '.join(line) + '\n')
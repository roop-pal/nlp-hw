word_counts = {}
train = open('ner_train.dat', 'r')
for line in train:
    line = line.strip('\n').split(' ')
    if len(line[0]) > 0:
        if line[0] not in word_counts:
            word_counts[line[0]] = 0
        word_counts[line[0]] += 1
train.close()


def classify(s):
    if s.isdigit():
        return '_DIGIT_'
    for i in s:
        if i.isdigit():
            return '_PARTNUM_'
    if all([i == '-' for i in s]):
        return '_DASHES_'
    s = s.replace('-','')
    s = s.replace('.','')
    if s.isupper() and len(s) < 4:
        return '_ABBREV_'
    if s[0].isupper() and s[1:].islower():
        return '_PROPER_'
    return '_RARE_'


train = open('ner_train.dat', 'r')
with open('q6_ner_train.dat', 'w') as outfile:
    for line in train:
        line = line.strip('\n').split(' ')
        if len(line[0]) > 0:
            if word_counts[line[0]] < 5:
                line[0] = classify(line[0])
        outfile.write(' '.join(line) + '\n')
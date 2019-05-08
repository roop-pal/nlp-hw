import os,sys
from decoder import Decoder
import dynet as dynet
import random
import matplotlib.pyplot as plt


def load_file(filename):
    d = {}
    for line in open(filename, 'r'):
        line = line.strip('\n').split(' ')
        d[line[0]] = int(line[1])
    return d.keys(), d


class Vocab:
    def __init__(self):
        self.words, self.words_dict = load_file('data/vocabs.word')
        self.pos, self.pos_dict = load_file('data/vocabs.pos')
        self.labels, self.labels_dict = load_file('data/vocabs.labels')
        self.actions, self.actions_dict = load_file('data/vocabs.actions')

    def word2id(self, word):
        return self.words_dict[word] if word in self.words_dict else self.words_dict['<unk>']

    def pos2id(self, pos):
        return self.pos_dict[pos] if pos in self.pos_dict else self.pos_dict['<null>']

    def label2id(self, label):
        return self.labels_dict[label]

    def action2id(self, action):
        return self.actions_dict[action]

    def num_actions(self):
        return len(self.actions_dict.keys())


class Network:
    def __init__(self, vocab, minibatch_size=1000, hidden_dim=200, dropout=False):
        self.vocab = vocab
        self.minibatch_size = minibatch_size
        self.dropout = dropout
        self.model = dynet.Model()
        self.updater = dynet.AdamTrainer(self.model)

        # create embeddings for words and tag features.
        word_embed_dim = 64
        pos_embed_dim = 32
        label_embed_dim = 32
        self.word_embedding = self.model.add_lookup_parameters((len(vocab.words), word_embed_dim))
        self.pos_embedding = self.model.add_lookup_parameters((len(vocab.pos), pos_embed_dim))
        self.label_embedding = self.model.add_lookup_parameters((len(vocab.labels), label_embed_dim))

        # assign transfer function
        self.transfer = dynet.rectify  # can be dynet.logistic or dynet.tanh as well.

        self.input_dim = 20 * (word_embed_dim + pos_embed_dim) + 12 * label_embed_dim
        self.hidden_layer1 = self.model.add_parameters((hidden_dim, self.input_dim))
        self.hidden_layer1_bias = self.model.add_parameters(hidden_dim, init=dynet.ConstInitializer(0.2))

        self.hidden_layer2 = self.model.add_parameters((hidden_dim, hidden_dim))
        self.hidden_layer2_bias = self.model.add_parameters(hidden_dim, init=dynet.ConstInitializer(0.2))

        # define the output weight.
        actions_dim = vocab.num_actions()
        self.output_layer = self.model.add_parameters((actions_dim, hidden_dim))
        self.output_bias = self.model.add_parameters(actions_dim, init=dynet.ConstInitializer(0))

    def build_graph(self, features):
        # extract word and tags ids
        word_ids = [self.vocab.word2id(word_feat) for word_feat in features[:20]]
        pos_ids = [self.vocab.pos2id(pos_feat) for pos_feat in features[20:40]]
        label_ids = [self.vocab.label2id(label_feat) for label_feat in features[40:]]

        # extract word embeddings and tag embeddings from features
        word_embeds = [self.word_embedding[wid] for wid in word_ids]
        pos_embeds = [self.pos_embedding[tid] for tid in pos_ids]
        label_embeds = [self.label_embedding[lid] for lid in label_ids]

        # concatenating all features (recall that '+' for lists is equivalent to appending two lists)
        embedding_layer = dynet.concatenate(word_embeds + pos_embeds + label_embeds)

        # calculating the hidden layer
        # .expr() converts a parameter to a matrix expression in dynet (its a dynet-specific syntax).
        hidden1 = self.transfer(self.hidden_layer1.expr() * embedding_layer + self.hidden_layer1_bias.expr())

        if self.dropout:
            hidden1 = dynet.dropout(hidden1, 0.2)

        hidden2 = self.transfer(self.hidden_layer2 * hidden1 + self.hidden_layer2_bias)

        # calculating the output layer
        output = self.output_layer.expr() * hidden2 + self.output_bias.expr()

        # return the output as a dynet vector (expression)
        return output

    def train(self, train_file, epochs=7):
        # matplotlib config
        loss_values = []
        plt.ion()
        ax = plt.gca()
        ax.set_xlim([0, 10])
        ax.set_ylim([0, 3])
        plt.title("Loss over time")
        plt.xlabel("Minibatch")
        plt.ylabel("Loss")

        for i in range(epochs):
            print('started epoch', (i+1))
            losses = []
            train_data = open(train_file, 'r').read().strip().split('\n')

            # shuffle the training data.
            random.shuffle(train_data)

            step = 0
            for line in train_data:
                fields = line.strip().split(' ')
                features, label = fields[:-1], self.vocab.action2id(fields[-1])
                result = self.build_graph(features)
                loss = dynet.pickneglogsoftmax(result, label)
                losses.append(loss)
                step += 1

                if len(losses) >= self.minibatch_size:
                    # now we have enough loss values to get loss for minibatch
                    minibatch_loss = dynet.esum(losses) / len(losses)

                    # calling dynet to run forward computation for all minibatch items
                    minibatch_loss.forward()

                    # getting float value of the loss for current minibatch
                    minibatch_loss_value = minibatch_loss.value()

                    # printing info and plotting
                    loss_values.append(minibatch_loss_value)
                    if len(loss_values)%10==0:
                        ax.set_xlim([0, len(loss_values)+10])
                        ax.plot(loss_values)
                        plt.draw()
                        plt.pause(0.0001)
                        progress = round(100 * float(step) / len(train_data), 2)
                        print('current minibatch loss', minibatch_loss_value, 'progress:', progress, '%')

                    # calling dynet to run backpropagation
                    minibatch_loss.backward()

                    # calling dynet to change parameter values with respect to current backpropagation
                    self.updater.update()

                    # empty the loss vector
                    losses = []

                    # refresh the memory of dynet
                    dynet.renew_cg()

            # there are still some minibatch items in the memory but they are smaller than the minibatch size
            # so we ask dynet to forget them
            dynet.renew_cg()


class DepModel:
    def __init__(self, part):
        '''
            You can add more arguments for examples actions and model paths.
            You need to load your model here.
            actions: provides indices for actions.
            it has the same order as the data/vocabs.actions file.
        '''
        # if you prefer to have your own index for actions, change this.
        self.actions = ['SHIFT', 'LEFT-ARC:rroot', 'LEFT-ARC:cc', 'LEFT-ARC:number', 'LEFT-ARC:ccomp', 'LEFT-ARC:possessive', 'LEFT-ARC:prt', 'LEFT-ARC:num', 'LEFT-ARC:nsubjpass', 'LEFT-ARC:csubj', 'LEFT-ARC:conj', 'LEFT-ARC:dobj', 'LEFT-ARC:nn', 'LEFT-ARC:neg', 'LEFT-ARC:discourse', 'LEFT-ARC:mark', 'LEFT-ARC:auxpass', 'LEFT-ARC:infmod', 'LEFT-ARC:mwe', 'LEFT-ARC:advcl', 'LEFT-ARC:aux', 'LEFT-ARC:prep', 'LEFT-ARC:parataxis', 'LEFT-ARC:nsubj', 'LEFT-ARC:<null>', 'LEFT-ARC:rcmod', 'LEFT-ARC:advmod', 'LEFT-ARC:punct', 'LEFT-ARC:quantmod', 'LEFT-ARC:tmod', 'LEFT-ARC:acomp', 'LEFT-ARC:pcomp', 'LEFT-ARC:poss', 'LEFT-ARC:npadvmod', 'LEFT-ARC:xcomp', 'LEFT-ARC:cop', 'LEFT-ARC:partmod', 'LEFT-ARC:dep', 'LEFT-ARC:appos', 'LEFT-ARC:det', 'LEFT-ARC:amod', 'LEFT-ARC:pobj', 'LEFT-ARC:iobj', 'LEFT-ARC:expl', 'LEFT-ARC:predet', 'LEFT-ARC:preconj', 'LEFT-ARC:root', 'RIGHT-ARC:rroot', 'RIGHT-ARC:cc', 'RIGHT-ARC:number', 'RIGHT-ARC:ccomp', 'RIGHT-ARC:possessive', 'RIGHT-ARC:prt', 'RIGHT-ARC:num', 'RIGHT-ARC:nsubjpass', 'RIGHT-ARC:csubj', 'RIGHT-ARC:conj', 'RIGHT-ARC:dobj', 'RIGHT-ARC:nn', 'RIGHT-ARC:neg', 'RIGHT-ARC:discourse', 'RIGHT-ARC:mark', 'RIGHT-ARC:auxpass', 'RIGHT-ARC:infmod', 'RIGHT-ARC:mwe', 'RIGHT-ARC:advcl', 'RIGHT-ARC:aux', 'RIGHT-ARC:prep', 'RIGHT-ARC:parataxis', 'RIGHT-ARC:nsubj', 'RIGHT-ARC:<null>', 'RIGHT-ARC:rcmod', 'RIGHT-ARC:advmod', 'RIGHT-ARC:punct', 'RIGHT-ARC:quantmod', 'RIGHT-ARC:tmod', 'RIGHT-ARC:acomp', 'RIGHT-ARC:pcomp', 'RIGHT-ARC:poss', 'RIGHT-ARC:npadvmod', 'RIGHT-ARC:xcomp', 'RIGHT-ARC:cop', 'RIGHT-ARC:partmod', 'RIGHT-ARC:dep', 'RIGHT-ARC:appos', 'RIGHT-ARC:det', 'RIGHT-ARC:amod', 'RIGHT-ARC:pobj', 'RIGHT-ARC:iobj', 'RIGHT-ARC:expl', 'RIGHT-ARC:predet', 'RIGHT-ARC:preconj', 'RIGHT-ARC:root']

        vocab = Vocab()
        if part == 1:
            self.network = Network(vocab)
        elif part == 2:
            self.network = Network(vocab, hidden_dim=400)
        elif part == 3:
            self.network = Network(vocab, dropout=True)
        self.network.train('data/train.data')
        # write your code here for additional parameters.
        # feel free to add more arguments to the initializer.

    def score(self, str_features):
        '''
        :param str_features: String features
        20 first: words, next 20: pos, next 12: dependency labels.
        DO NOT ADD ANY ARGUMENTS TO THIS FUNCTION.
        :return: list of scores
        '''
        # change this part of the code.
        return self.network.build_graph(str_features).value()


if __name__ == '__main__':
    part = int(os.path.abspath(sys.argv[1]))
    m = DepModel(part)
    input_p = os.path.abspath(sys.argv[2])
    output_p = os.path.abspath(sys.argv[3])
    Decoder(m.score, m.actions).parse(input_p, output_p)
    Decoder(m.score, m.actions).parse(input_p.replace('dev','test'), output_p.replace('dev','test'))
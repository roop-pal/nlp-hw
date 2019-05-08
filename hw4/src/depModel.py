import os,sys
from decoder import *
import dynet as dynet
import random
import matplotlib.pyplot as plt
import numpy as np

class Network:
    def __init__(self, vocab, properties):
        self.properties = properties
        self.vocab = vocab

        # first initialize a computation graph container (or model).
        self.model = dynet.Model()

        # assign the algorithm for backpropagation updates.
        self.updater = dynet.AdamTrainer(self.model)

        # create embeddings for words and tag features.
        self.word_embedding = self.model.add_lookup_parameters((vocab.num_words(), properties.word_embed_dim))
        self.tag_embedding = self.model.add_lookup_parameters((vocab.num_tag_feats(), properties.pos_embed_dim))

        # assign transfer function
        self.transfer = dynet.rectify  # can be dynet.logistic or dynet.tanh as well.

        # define the input dimension for the embedding layer.
        # here we assume to see two words after and before and current word (meaning 5 word embeddings)
        # and to see the last two predicted tags (meaning two tag embeddings)
        self.input_dim = 5 * properties.word_embed_dim + 2 * properties.pos_embed_dim

        # define the hidden layer.
        self.hidden_layer = self.model.add_parameters((properties.hidden_dim, self.input_dim))

        # define the hidden layer bias term and initialize it as constant 0.2.
        self.hidden_layer_bias = self.model.add_parameters(properties.hidden_dim, init=dynet.ConstInitializer(0.2))

        # define the output weight.
        self.output_layer = self.model.add_parameters((vocab.num_tags(), properties.hidden_dim))

        # define the bias vector and initialize it as zero.
        self.output_bias = self.model.add_parameters(vocab.num_tags(), init=dynet.ConstInitializer(0))

    def build_graph(self, features):
        # extract word and tags ids
        word_ids = [self.vocab.word2id(word_feat) for word_feat in features[0:5]]
        tag_ids = [self.vocab.feat_tag2id(tag_feat) for tag_feat in features[5:]]

        # extract word embeddings and tag embeddings from features
        word_embeds = [self.word_embedding[wid] for wid in word_ids]
        tag_embeds = [self.tag_embedding[tid] for tid in tag_ids]

        # concatenating all features (recall that '+' for lists is equivalent to appending two lists)
        embedding_layer = dynet.concatenate(word_embeds + tag_embeds)

        # calculating the hidden layer
        # .expr() converts a parameter to a matrix expression in dynet (its a dynet-specific syntax).
        hidden = self.transfer(self.hidden_layer.expr() * embedding_layer + self.hidden_layer_bias.expr())

        # calculating the output layer
        output = self.output_layer.expr() * hidden + self.output_bias.expr()

        # return the output as a dynet vector (expression)
        return output

    def train(self, train_file, epochs):
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
                features, label = fields[:-1], self.vocab.action_to_id(fields[-1])


                # gold_label = self.vocab.tag2id(label)
                result = self.build_graph(features)

                # getting loss with respect to negative log softmax function and the gold label.
                loss = dynet.pickneglogsoftmax(result, gold_label)

                # appending to the minibatch losses
                losses.append(loss)
                step += 1

                if len(losses) >= self.properties.minibatch_size:
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
                        print 'current minibatch loss', minibatch_loss_value, 'progress:', progress, '%'

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
    def __init__(self):
        '''
            You can add more arguments for examples actions and model paths.
            You need to load your model here.
            actions: provides indices for actions.
            it has the same order as the data/vocabs.actions file.
        '''
        # if you prefer to have your own index for actions, change this.
        self.actions = ['SHIFT', 'LEFT-ARC:rroot', 'LEFT-ARC:cc', 'LEFT-ARC:number', 'LEFT-ARC:ccomp', 'LEFT-ARC:possessive', 'LEFT-ARC:prt', 'LEFT-ARC:num', 'LEFT-ARC:nsubjpass', 'LEFT-ARC:csubj', 'LEFT-ARC:conj', 'LEFT-ARC:dobj', 'LEFT-ARC:nn', 'LEFT-ARC:neg', 'LEFT-ARC:discourse', 'LEFT-ARC:mark', 'LEFT-ARC:auxpass', 'LEFT-ARC:infmod', 'LEFT-ARC:mwe', 'LEFT-ARC:advcl', 'LEFT-ARC:aux', 'LEFT-ARC:prep', 'LEFT-ARC:parataxis', 'LEFT-ARC:nsubj', 'LEFT-ARC:<null>', 'LEFT-ARC:rcmod', 'LEFT-ARC:advmod', 'LEFT-ARC:punct', 'LEFT-ARC:quantmod', 'LEFT-ARC:tmod', 'LEFT-ARC:acomp', 'LEFT-ARC:pcomp', 'LEFT-ARC:poss', 'LEFT-ARC:npadvmod', 'LEFT-ARC:xcomp', 'LEFT-ARC:cop', 'LEFT-ARC:partmod', 'LEFT-ARC:dep', 'LEFT-ARC:appos', 'LEFT-ARC:det', 'LEFT-ARC:amod', 'LEFT-ARC:pobj', 'LEFT-ARC:iobj', 'LEFT-ARC:expl', 'LEFT-ARC:predet', 'LEFT-ARC:preconj', 'LEFT-ARC:root', 'RIGHT-ARC:rroot', 'RIGHT-ARC:cc', 'RIGHT-ARC:number', 'RIGHT-ARC:ccomp', 'RIGHT-ARC:possessive', 'RIGHT-ARC:prt', 'RIGHT-ARC:num', 'RIGHT-ARC:nsubjpass', 'RIGHT-ARC:csubj', 'RIGHT-ARC:conj', 'RIGHT-ARC:dobj', 'RIGHT-ARC:nn', 'RIGHT-ARC:neg', 'RIGHT-ARC:discourse', 'RIGHT-ARC:mark', 'RIGHT-ARC:auxpass', 'RIGHT-ARC:infmod', 'RIGHT-ARC:mwe', 'RIGHT-ARC:advcl', 'RIGHT-ARC:aux', 'RIGHT-ARC:prep', 'RIGHT-ARC:parataxis', 'RIGHT-ARC:nsubj', 'RIGHT-ARC:<null>', 'RIGHT-ARC:rcmod', 'RIGHT-ARC:advmod', 'RIGHT-ARC:punct', 'RIGHT-ARC:quantmod', 'RIGHT-ARC:tmod', 'RIGHT-ARC:acomp', 'RIGHT-ARC:pcomp', 'RIGHT-ARC:poss', 'RIGHT-ARC:npadvmod', 'RIGHT-ARC:xcomp', 'RIGHT-ARC:cop', 'RIGHT-ARC:partmod', 'RIGHT-ARC:dep', 'RIGHT-ARC:appos', 'RIGHT-ARC:det', 'RIGHT-ARC:amod', 'RIGHT-ARC:pobj', 'RIGHT-ARC:iobj', 'RIGHT-ARC:expl', 'RIGHT-ARC:predet', 'RIGHT-ARC:preconj', 'RIGHT-ARC:root']
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
        return [0]*len(self.actions)

if __name__=='__main__':
    m = DepModel()
    input_p = os.path.abspath(sys.argv[1])
    output_p = os.path.abspath(sys.argv[2])
    Decoder(m.score, m.actions).parse(input_p, output_p)
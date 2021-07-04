''' Data Loader class for training iteration '''
import random
import numpy as np
import torch
from torch.autograd import Variable
import Constants
import logging
import pickle
import os
import networkx as nx


class Options(object):
    def __init__(self):
        self.train_data = 'data_small/train.txt'
        self.test = 'data_small/test.csv'
        self.test_data = 'data_small/test.txt'

        self.u2idx_dict = 'data_small/u2idx.pickle'
        self.idx2u_dict = 'data_small/idx2u.pickle'

        self.graph = 'data_small/graph.txt'

        self.save_path = ''


class DataLoader(object):
    def __init__(self, data=0, load_dict=False, cuda=True, batch_size=256, maxlen=30, shuffle=True, test=False, with_EOS=False):
        self.options = Options()
        self.batch_size = batch_size
        self.maxlen = maxlen
        self.u2idx = {}
        self.idx2u = []
        self.data = data
        self.test = test
        self.with_EOS = with_EOS
        self.cuda = cuda

        if not load_dict:
            self.buildIndex()
            with open(self.options.u2idx_dict, 'wb') as handle:
                pickle.dump(self.u2idx, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(self.options.idx2u_dict, 'wb') as handle:
                pickle.dump(self.idx2u, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(self.options.u2idx_dict, 'rb') as handle:
                self.u2idx = pickle.load(handle)
            with open(self.options.idx2u_dict, 'rb') as handle:
                self.idx2u = pickle.load(handle)

        self.user_size = len(self.u2idx)

        self.train_cascades, train_len = self.readFromFile(self.options.train_data)
        self.test_cascades, test_len = self.readFromFile(self.options.test_data)
        self.train_size = len(self.train_cascades)
        self.test_size = len(self.test_cascades)
        print("training set size:%d\ntesting set size:%d" % (self.train_size, self.test_size))
        print("total set size:%d" % (self.train_size + self.test_size))
        print("average length of cascades:%f" % ((train_len + test_len + 0.0) / (self.train_size + self.test_size)))  # 级联平均长度

        if self.data == 0:  # train data
            self.n_batch = int(np.ceil(len(self.train_cascades) / batch_size))  # train batch数量
        else:  # test data
            self.n_batch = int(np.ceil(len(self.test_cascades) / batch_size))

        self.iter_count = 0
        self.need_shuffle = shuffle
        if self.need_shuffle:
            random.shuffle(self.train_cascades)

        self.G = self.load_graph(self.options.graph)
        print('number of nodes:%d' % (self.G.number_of_nodes()))
        print('number of edges:%d' % (self.G.number_of_edges()))
        print('average degree:%f' % (self.G.number_of_edges() / self.G.number_of_nodes()))

        self.train_graph = self.load_structural_context(self.train_cascades)
        self.test_graph = self.load_structural_context(self.test_cascades)
        print("training set size:%d   testing set size:%d" % (len(self.train_graph), len(self.test_graph)))

    def load_graph(self, filename):
        G = nx.Graph()
        n_nodes = len(self.u2idx)
        G.add_nodes_from(range(n_nodes))
        with open(filename, 'r') as f:
            for line in f:
                u, v = line.strip().split()
                if (u in self.u2idx) and (v in self.u2idx):
                    u = self.u2idx[u]
                    v = self.u2idx[v]
                    G.add_edge(u, v)
        return G

    def buildIndex(self):
        # compute an index of the users that appear at least once in the training and testing cascades.
        opts = self.options

        train_user_set = set()
        test_user_set = set()

        lineid = 0
        for line in open(opts.train_data):
            lineid += 1
            if len(line.strip()) == 0:
                continue
            query, cascade = line.strip().split(' ', 1)
            sequence = [query] + cascade.split(' ')[::2]
            sequence = sequence[:self.maxlen]
            for user in sequence:
                train_user_set.add(user)

        for line in open(opts.test):
            if len(line.strip()) == 0:
                continue
            sequence = line.strip().split(',')
            for user in sequence:
                test_user_set.add(user)

        user_set = train_user_set | test_user_set

        pos = 0
        for user in user_set:
            self.u2idx[user] = pos
            self.idx2u.append(user)
            pos += 1
        opts.user_size = len(user_set)
        self.user_size = len(user_set)
        print("user_size : %d" % (opts.user_size))

    def readFromFile(self, filename):
        """read all cascade from training or testing files. """
        total_len = 0
        t_cascades = []
        for line in open(filename):
            if len(line.strip()) == 0:
                continue
            query, cascade = line.strip().split(' ', 1)
            userlist = []
            sequence = [query] + cascade.split(' ')[::2]
            for user in sequence:
                if user in self.u2idx:
                    userlist.append(self.u2idx[user])
            total_len += len(userlist)
            if self.with_EOS:
                userlist.append(Constants.EOS)
            t_cascades.append(userlist)
        return t_cascades, total_len

    def load_structural_context(self, cascades):
        examples = []
        for cascade in cascades:
            length = len(cascade)
            source = []
            target = []
            for i, node in enumerate(cascade):
                if i == 0:
                    continue
                prefix = cascade[: i + 1]
                #if len(prefix) > 5:
                #    prefix = prefix[-5:]
                for x in prefix:
                    if (x, node) in self.G.edges:
                        source.append(x)
                        target.append(node)

            edge_index = [source, target]
            examples.append(edge_index)
        return examples

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        return self.n_batch

    def next(self):
        ''' Get the next batch '''
        def seq_pad_to_longest(insts):  # 加入padding的序列数据
            ''' Pad the instance to the max seq length in batch '''

            max_len = max(len(inst) for inst in insts)

            inst_data = np.array([
                inst + [Constants.PAD] * (max_len - len(inst))
                for inst in insts])

            inst_data_tensor = Variable(
                torch.LongTensor(inst_data), volatile=self.test)

            if self.cuda:
                inst_data_tensor = inst_data_tensor.cuda()

            return inst_data_tensor

        def graph_pad_to_longest(insts):
            max_len = max(len(inst[0]) for inst in insts)
            inst_data = np.array([
                [inst[0] + [Constants.PAD] * (max_len - len(inst[0])),
                inst[0] + [Constants.PAD] * (max_len - len(inst[1]))]
                for inst in insts])
            inst_data_tensor = Variable(
                torch.LongTensor(inst_data), volatile=self.test)
            if self.cuda:
                inst_data_tensor = inst_data_tensor.cuda()
            return inst_data_tensor

        if self.iter_count < self.n_batch:
            batch_idx = self.iter_count
            self.iter_count += 1

            start_idx = batch_idx * self.batch_size
            end_idx = (batch_idx + 1) * self.batch_size

            if self.data == 0:
                seq_insts = self.train_cascades[start_idx:end_idx]
                graph_insts = self.train_graph[start_idx:end_idx]
            else:
                seq_insts = self.test_cascades[start_idx:end_idx]
                graph_insts = self.test_graph[start_idx:end_idx]
            seq_data = seq_pad_to_longest(seq_insts)
            graph_data = graph_pad_to_longest(graph_insts)

            return seq_data, graph_data
        else:
            if self.need_shuffle:
                random.shuffle(self.train_cascades)

            self.iter_count = 0
            raise StopIteration()


class Loader_GAT:
    def __init__(self, data, batch_size,shuffle_data=False):
        self.batch_size = batch_size
        self.idx = 0
        self.data = data
        self.shuffle = shuffle_data
        self.n = len(data)
        self.indices = np.arange(self.n, dtype="int32")

    def __len__(self):
        return len(self.data) // self.batch_size + 1

    def __call__(self):
        if self.shuffle and self.idx == 0:
            np.random.shuffle(self.indices)

        batch_indices = self.indices[self.idx: self.idx + self.batch_size]
        batch_examples = [self.data[i] for i in batch_indices]

        self.idx += self.batch_size
        if self.idx >= self.n:
            self.idx = 0

        labels = [t['label'] for t in batch_examples]
        labels_vector = np.array(labels)
        edge_index = [t['graph'] for t in batch_examples]

        return (edge_index, labels_vector)
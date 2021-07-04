''' Data Loader class for training iteration '''
import random
import numpy as np
import torch
from torch.autograd import Variable
import Constants
import logging
import pickle


class Options(object):
    def __init__(self):
        self.train_data = 'data_small/train.txt'
        self.test_data = 'data_small/test.csv'
        self.valid_data = 'data_small/test.txt'

        self.u2idx_dict = 'data_small/u2idx.pickle'
        self.idx2u_dict = 'data_small/idx2u.pickle'

        self.save_path = ''


class DataLoader(object):
    def __init__(
            self, data=0, load_dict=False, cuda=True, batch_size=256, shuffle=True, test=False,
            with_EOS=True):
        self.options = Options()
        self.options.batch_size = batch_size
        self._u2idx = {}
        self._idx2u = []
        self.data = data
        self.test = test
        self.with_EOS = with_EOS

        if not load_dict:
            self._buildIndex()
            with open(self.options.u2idx_dict, 'wb') as handle:
                pickle.dump(self._u2idx, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(self.options.idx2u_dict, 'wb') as handle:
                pickle.dump(self._idx2u, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(self.options.u2idx_dict, 'rb') as handle:
                self._u2idx = pickle.load(handle)
            with open(self.options.idx2u_dict, 'rb') as handle:
                self._idx2u = pickle.load(handle)
            self.user_size = len(self._u2idx)

        self._train_cascades, train_len = self._readFromFile(self.options.train_data)
        self._test_cascades, test_len = self._readFromFile(self.options.valid_data)
        self.train_size = len(self._train_cascades)
        self.test_size = len(self._test_cascades)
        print("training set size:%d   testing set size:%d" % (self.train_size, self.test_size))
        print(self.train_size + self.test_size)
        print((train_len + test_len + 0.0) / (self.train_size + self.test_size))  # 级联平均长度

        self.cuda = cuda

        if self.data == 0:
            self._n_batch = int(np.ceil(len(self._train_cascades) / batch_size))  # train batch数量
        else:
            self._n_batch = int(np.ceil(len(self._test_cascades) / batch_size))

        self._batch_size = self.options.batch_size
        self.iter_count = 0
        self._need_shuffle = shuffle
        if self._need_shuffle:
            random.shuffle(self._train_cascades)

    def _buildIndex(self):
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
            sequence = sequence[:30]
            for user in sequence:
                train_user_set.add(user)

        for line in open(opts.test_data):
            if len(line.strip()) == 0:
                continue
            sequence = line.strip().split(',')
            for user in sequence:
                test_user_set.add(user)

        user_set = train_user_set | test_user_set

        pos = 0
        for user in user_set:
            self._u2idx[user] = pos
            self._idx2u.append(user)
            pos += 1
        opts.user_size = len(user_set)
        self.user_size = len(user_set)
        print("user_size : %d" % (opts.user_size))

    def _readFromFile(self, filename):
        """read all cascade from training or testing files. """
        total_len = 0
        t_cascades = []
        i = 0
        for line in open(filename):
            if i > 10:
                break
            i += 1
            if len(line.strip()) == 0:
                continue
            query, cascade = line.strip().split(' ', 1)
            userlist = []
            sequence = [query] + cascade.split(' ')[::2]
            for user in sequence:
                if user in self._u2idx:
                    userlist.append(self._u2idx[user])
            if 1 < len(userlist) <= 500:
                total_len += len(userlist)
                if self.with_EOS:
                    userlist.append(Constants.EOS)
                t_cascades.append(userlist)
        return t_cascades, total_len

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        return self._n_batch

    def next(self):
        ''' Get the next batch '''

        def pad_to_longest(insts):
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

        if self.iter_count < self._n_batch:
            batch_idx = self.iter_count
            self.iter_count += 1

            start_idx = batch_idx * self._batch_size
            end_idx = (batch_idx + 1) * self._batch_size

            if self.data == 0:
                seq_insts = self._train_cascades[start_idx:end_idx]
            else:
                seq_insts = self._test_cascades[start_idx:end_idx]
            seq_data = pad_to_longest(seq_insts)

            return seq_data
        else:

            if self._need_shuffle:
                random.shuffle(self._train_cascades)

            self.iter_count = 0
            raise StopIteration()


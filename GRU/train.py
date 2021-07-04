'''
This script handling the training process.
'''

import argparse
import math
import time
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import Constants
from model import RNNModel
from Optim import ScheduledOptim
from DataLoader import DataLoader


def get_performance(crit, pred, gold):
    loss = crit(pred, gold.contiguous().view(-1))
    pred = pred.max(1)[1]

    gold = gold.contiguous().view(-1)
    n_correct = pred.data.eq(gold.data)
    n_correct = n_correct.masked_select(gold.ne(Constants.PAD).data).sum().float()
    return loss, n_correct


def train_epoch(model, training_data, crit, optimizer):

    model.train()

    total_loss = 0.0
    n_total_words = 0.0
    n_total_correct = 0.0

    batch_num = 0.0

    for batch in tqdm(
            training_data, mininterval=2,
            desc='  - (Training)   ', leave=False):

        # prepare data
        tgt = batch
        gold = tgt[:, 1:]
        n_words = gold.data.ne(Constants.PAD).sum().float()
        n_total_words += n_words

        batch_num += tgt.size(0)

        optimizer.zero_grad()
        pred, *_ = model(tgt)

        # backward
        loss, n_correct = get_performance(crit, pred, gold)
        loss.backward()

        # update parameters
        optimizer.step()
        optimizer.update_learning_rate()

        # note keeping
        n_total_correct += n_correct
        total_loss += loss.item()

    return total_loss/n_total_words, n_total_correct/n_total_words


def train(model, training_data, test_data, crit, optimizer, opt):

    train_accus = []
    for epoch_i in range(opt.epoch):
        print('[ Epoch', epoch_i, ']')

        start = time.time()
        train_loss, train_accu = train_epoch(model, training_data, crit, optimizer)
        print('  - (Training)   ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, '\
              'elapse: {elapse:3.3f} min'.format(
                  ppl=math.exp(min(train_loss, 100)), accu=100*train_accu,
                  elapse=(time.time()-start)/60))
        train_accus.append(train_accu)

        model_state_dict = model.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'settings': opt,
            'epoch': epoch_i}

        if opt.save_model:
            if opt.save_mode == 'all':
                model_name = opt.save_model + '_accu_{accu:3.3f}.chkpt'.format(accu=100*valid_accu)
                torch.save(checkpoint, model_name)
            elif opt.save_mode == 'best':
                model_name = opt.save_model + '.pth'
                if train_accu >= max(train_accus):
                    torch.save(checkpoint, model_name)
                    print('    - [Info] The checkpoint file has been updated.')


def main():
    torch.set_num_threads(4)
    ''' Main function'''
    parser = argparse.ArgumentParser()

    parser.add_argument('-epoch', type=int, default=30)
    parser.add_argument('-batch_size', type=int, default=16)

    parser.add_argument('-d_model', type=int, default=64)
    parser.add_argument('-d_inner_hid', type=int, default=64)
    parser.add_argument('-n_warmup_steps', type=int, default=1000)

    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-embs_share_weight', action='store_true')
    parser.add_argument('-proj_share_weight', action='store_true')
    parser.add_argument('-pos_emb', type=int, default=1)

    parser.add_argument('-save_model', default='model')
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')
    parser.add_argument('-no_cuda', action='store_true')

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda
    opt.d_word_vec = opt.d_model
    if opt.pos_emb == 1:
        opt.pos_emb = True
    else:
        opt.pos_emb = False
    

    #========= Preparing DataLoader =========#
    train_data = DataLoader(data=0, load_dict=False, batch_size=opt.batch_size, cuda=opt.cuda)
    test_data = DataLoader(data=1, batch_size=opt.batch_size, cuda=opt.cuda)

    opt.user_size = train_data.user_size

    #========= Preparing Model =========#
    RLLearner = RNNModel('GRUCell', opt)

    optimizer = ScheduledOptim(
        optim.Adam(
            RLLearner.parameters(),
            betas=(0.9, 0.98), eps=1e-09),
        opt.d_model, opt.n_warmup_steps)

    def get_criterion(user_size):
        weight = torch.ones(user_size)
        weight[Constants.PAD] = 0
        weight[Constants.EOS] = 1
        return nn.CrossEntropyLoss(weight, size_average=False)
    crit = get_criterion(train_data.user_size)

    if opt.cuda:
        RLLearner = RLLearner.cuda()
        crit = crit.cuda()

    train(RLLearner, train_data, test_data, crit, optimizer, opt)

if __name__ == '__main__':
    main()

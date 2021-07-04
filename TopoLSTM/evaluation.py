import numpy as np


gold_dir = 'data_small/test.txt'
pred_dir = 'answer/answer.csv'
begin = 'data_small/test.csv'

test_seq = []
with open(begin, 'r') as f:
    for line in f:
        sequence = line.strip().split(',')
        test_seq.append(sequence)

gold_seq = []
with open(gold_dir, 'r') as f:
    for line in f:
        query, cascade = line.strip().split(' ', 1)
        sequence = [query] + cascade.split(' ')[::2]
        if sequence[:4] in test_seq:
            gold_seq.append(sequence[4:])

pred_seq = []
with open(pred_dir, 'r') as f:
    for line in f:
        sequence = line.strip().split(',')
        pred_seq.append(sequence[4:])

def hits_k(y_prob, y, k=10):
    acc = []
    for i in range(len(y)):
        top_k = y_prob[i][:k]
        for gold in y[i]:
            acc += [1. if gold in top_k else 0.]
    return sum(acc) / len(acc)

def mapk(y_prob, y, k=10):
    predicted = y_prob
    actual = y
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])

def apk(actual, predicted, k=10):
    if len(predicted) > k:
        predicted = predicted[:k]
    score = 0.0
    num_hits = 0.0
    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    return score / min(len(actual), k)


k_list = [10, 50, 100]
y_prob = pred_seq
y = gold_seq

scores = {}
for k in k_list:
    scores['hits@' + str(k)] = hits_k(y_prob, y, k=k)
    scores['map@' + str(k)] = mapk(y_prob, y, k=k)
print(scores)

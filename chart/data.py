import pandas as pd
import numpy as np
import torch

class Index:

    def __init__(self, dim, bound):
        self._idx = [0 for i in range(dim)]
        self._idx[-1] = -1
        self.dim = dim
        self.bound = bound
        print('bound', self.bound)

    def setidx(self, *args):
        # import ipdb; ipdb.set_trace()
        self._idx = list(args)

    def add(self, d=-1):
        assert isinstance(d, int)
        if d < 0:
            d = self.dim + d
        self._idx[d] += 1
        i = 0
        for i in range(d,0,-1):
            if self._idx[i] >= self.bound[i]:
                self._idx[i] = 0
                self._idx[i-1] += 1
        if i == 0:
            if self._idx[0] >= self.bound[0]:
                self._idx[0] = 0
        return self._idx

    def sub(self, d=-1):
        assert isinstance(d, int)
        if d < 0:
            d = self.dim + d
        self._idx[d] -= 1
        i = 0
        for i in range(d,0,-1):
            if self._idx[i] < 0:
                self._idx[i] = self.bound[i] - 1
                self._idx[i-1] -= 1
        if i == 0:
            if self._idx[0] < 0:
                self._idx[0] = self.bound[0] - 1
        return self._idx


def legalize(s):
    return s.replace('<', '[').replace('>', ']').replace('▁','')

def legalize_lb(lb):
    if not lb[0].startswith('▁'):
        return lb
    res = []
    for i in lb:
        if i.startswith('▁'):
            res.append(i.replace('▁',''))
        else:
            res[-1] = res[-1] + '@@'
            res.append(i)
    return res

class MultiAttentionMeanDataGenerator:

    def __init__(self, data):
        self._data = data # l x h x ny x nx
        self.idx = -1

    def __len__(self):
        return len(self._data)

    def last(self):
        self.idx -= 1
        return self.__getitem__(self.idx)

    def next(self):
        self.idx += 1
        return self.__getitem__(self.idx)

    def __getitem__(self, x):
        while x < 0:
            x += len(self._data)
        if x >= len(self._data):
            x = x % len(self._data)
        self.idx = x
        data = self._data[x] # data: dict of weights [ny, nx]
        x_lb = legalize_lb(data['src'])
        y_lb = legalize_lb(data['tgt'])
        metrics = data['metrics']
        info = ''
        for k, v in metrics.items():
            info += '{}: {:.1f} '.format(k, v[0]*100)
        weights = {}
        # import ipdb; ipdb.set_trace()
        for k, v in data['weights'].items():
            if isinstance(v, list):
                # [(j, i, v)]
                tw = v
            else:
                tw = v
                tw = tw[:len(y_lb), :len(x_lb)]
                tw = [(j, i, tw[i,j].item()) for i in range(tw.shape[0]) for j in range(tw.shape[1])]
            weights[k] = [{'name': '[{}, {}]'.format(legalize(x_lb[x[0]]), legalize(y_lb[x[1]])), 'value': [x[0], x[1], x[2]*100]} for x in tw]
        value = {
            'info': info,
            'weights': weights
        }
        return x_lb, y_lb, value

if __name__ == "__main__":
    data = [["a", "b", [1]], ["c", "d", [2]], ["e", "f", [3]]]
    dataloader = DataGenerator(data)
    for d in dataloader:
        print(d)

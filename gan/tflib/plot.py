import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import collections
try:
    import cPickle as pickle
except ImportError:
    import pickle
import os

_since_beginning = collections.defaultdict(lambda: {})
_since_last_flush = collections.defaultdict(lambda: {})

_iter = [0]


def reset():
    _since_beginning.clear()
    _since_last_flush.clear()
    _iter[0] = 0


def tick():
    _iter[0] += 1


def plot(name, value):
    _since_last_flush[name][_iter[0]] = value


def flush(dir, filename='log.pkl'):
    prints = []

    log_folder = os.path.join(dir, 'log')
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    for name, vals in _since_last_flush.items():
        prints.append("{}\t{}".format(name, np.mean(vals.values())))
        _since_beginning[name].update(vals)

        x_vals = np.sort(_since_beginning[name].keys())
        y_vals = [_since_beginning[name][x] for x in x_vals]

        plt.clf()
        plt.plot(x_vals, y_vals)
        plt.xlabel('iteration')
        plt.ylabel(name)

        plt.savefig(os.path.join(log_folder, name.replace(' ', '_') + '.jpg'))

    print("iter {}\t{}".format(_iter[0], "\t".join(prints)))
    _since_last_flush.clear()

    with open(os.path.join(log_folder, filename), 'wb') as f:
        pickle.dump(dict(_since_beginning), f, pickle.HIGHEST_PROTOCOL)

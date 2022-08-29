import pandas as pd
import numpy as np
import tensorflow as tf
import csv


def make_summary(value_dict):
    return tf.Summary(value=[tf.Summary.Value(tag=k, simple_value=v) for k, v in value_dict.items()])


def load(data_file):
    df = pd.read_csv(data_file, sep='\t', names=['p1', 'p2', 'words', 'mentions', 'types'], quoting=csv.QUOTE_NONE)
    words = np.array(df.words)
    mentions = np.array(df.mentions)
    positions = np.array([[x, y] for x, y in zip(df.p1, df.p2)])
    labels = np.array(df.types)
    return words, mentions, positions, labels


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((data_size - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

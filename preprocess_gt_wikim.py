#generates preprocessed wikim with gt_file
import pandas as pd
from optparse import OptionParser
import os

import config
from collections import defaultdict

import pickle


def pkl_save(fname, data, protocol=-1):
    with open(fname, 'wb') as f:
        pickle.dump(data, f, protocol)


def pkl_load(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)


def path_count(types):
    cnt = 0
    for a in types:
        flag = 1
        for b in types:
            if len(a) >= len(b):
                continue
            if (a == b[:len(a)]) and (b[len(a)] == '/'):
                flag = 0
        cnt += flag
    return cnt


def read_out_type(types):
    out_type = []
    for a in types:
        flag = True
        for b in types:
            if len(a) >= len(b):
                continue
            if (a == b[:len(a)]) and (b[len(a)] == '/'):
                flag = False
        if flag:
            out_type.append(a)
    return out_type


def create_type_dict(infile, outfile, full_path):
    df = pd.read_csv(infile, sep='\t', names=['p1', 'p2', 'text', 'type', 'f'])
    size = df.shape[0]
    type_set = set()
    freq = defaultdict(int)
    for i in range(size):
        types = df['type'][i].split()
        out_type = read_out_type(types)
        if full_path:
            type_set.update(types)
        else:
            type_set.update(out_type)
        for t in types:
            freq[t] += 1

    type2id = {y: x for x, y in enumerate(type_set)}
    type_dict = defaultdict(list)
    for a in type2id.keys():
        for b in type2id.keys():
            if len(a) >= len(b):
                continue
            if (a == b[:len(a)]) and (b[len(a)] == '/'):
                type_dict[a].append(b)
    pkl_save(outfile, (type2id, type_dict))


def clear_text(text):
    text = text.replace('-LRB-', '``')
    text = text.replace('-RRB-', "''")
    text = text.replace('-LSB-', '[')
    text = text.replace('-RSB-', ']')
    text = text.replace('-LCB-', '{')
    text = text.replace('-RCB-', '}')

    return text.strip()


def preprocess(if_clean=False, full_path=False):

    raw_all_file = config.WIKIM_ALL
    raw_train_file = config.WIKIM_TRAIN
    raw_valid_file = config.WIKIM_VALID
    raw_test_file = config.WIKI_TEST
    clean_train_file = config.WIKIM_TRAIN_CLEAN
    clean_test_file = config.WIKIM_TEST_CLEAN
    clean_gt_file = config.WIKIM_GT_CLEAN
    type_file = config.WIKIM_TYPE

    if not os.path.exists(type_file):
        create_type_dict(raw_all_file, type_file, full_path)

    #train
    df = pd.read_csv(raw_train_file, sep='\t', names=['p1', 'p2', 'text', 'type', 'f'])
    # df_valid = pd.read_csv(raw_valid_file, sep='\t', names=['p1', 'p2', 'text', 'type', 'f'])
    # df = pd.concat((df_train, df_valid), ignore_index=True)
    # df = pd.concat((df_train, df_valid), ignore_index=True)
    size = df.shape[0]
    outfile = open(clean_train_file, 'w')
    for i in range(size):
        p1 = df['p1'][i]
        p2 = df['p2'][i]
        text = df['text'][i]
        types = df['type'][i].split()
        if (not path_count(types) == 1) and if_clean:
            continue

        text = clear_text(text)
        tokens = text.split()
        if p1 >= len(tokens):
            continue
        mention = ' '.join(tokens[p1:p2])

        if p1 == 0:
            mention = '<PAD> ' + mention
        else:
            mention = tokens[p1 - 1] + ' ' + mention
        if p2 >= len(tokens):
            mention = mention + ' <PAD>'
        else:
            mention = mention + ' ' + tokens[p2]

        offset = max(0, p1 - config.WINDOW_SIZE)
        text = ' '.join(tokens[offset:min(len(tokens), p2 + config.WINDOW_SIZE - 1)])
        p1 -= offset
        p2 -= offset

        out_type = read_out_type(types)

        if len(out_type) > 0:
            if full_path:
                outfile.write('%d\t%d\t%s\t%s\t%s\n' % (p1, p2, text, mention, ' '.join(types)))
            else:
                outfile.write('%d\t%d\t%s\t%s\t%s\n' % (p1, p2, text, mention, ' '.join(out_type)))
    outfile.close()

    #gt
    df = pd.read_csv(raw_valid_file, sep='\t', names=['p1', 'p2', 'text', 'type', 'f'])
    size = df.shape[0]
    outfile = open(clean_gt_file, 'w')
    for i in range(size):
        p1 = df['p1'][i]
        p2 = df['p2'][i]
        text = df['text'][i]
        types = df['type'][i].split()
        if (not path_count(types) == 1) and if_clean:
            continue

        text = clear_text(text)
        tokens = text.split()
        if p1 >= len(tokens):
            continue
        mention = ' '.join(tokens[p1:p2])

        if p1 == 0:
            mention = '<PAD> ' + mention
        else:
            mention = tokens[p1 - 1] + ' ' + mention
        if p2 >= len(tokens):
            mention = mention + ' <PAD>'
        else:
            mention = mention + ' ' + tokens[p2]

        offset = max(0, p1 - config.WINDOW_SIZE)
        text = ' '.join(tokens[offset:min(len(tokens), p2 + config.WINDOW_SIZE - 1)])
        p1 -= offset
        p2 -= offset

        out_type = read_out_type(types)

        if len(out_type) > 0:
            if full_path:
                outfile.write('%d\t%d\t%s\t%s\t%s\n' % (p1, p2, text, mention, ' '.join(types)))
            else:
                outfile.write('%d\t%d\t%s\t%s\t%s\n' % (p1, p2, text, mention, ' '.join(out_type)))
    outfile.close()


    df = pd.read_csv(raw_test_file, sep='\t', names=['p1', 'p2', 'text', 'type', 'f'])
    size = df.shape[0]
    outfile = open(clean_test_file, 'w')
    for i in range(size):
        p1 = df['p1'][i]
        p2 = df['p2'][i]
        text = df['text'][i]
        types = df['type'][i].split()

        text = clear_text(text)
        tokens = text.split()
        if p1 >= len(tokens):
            continue
        mention = ' '.join(tokens[p1:p2])

        if p1 == 0:
            mention = '<PAD> ' + mention
        else:
            mention = tokens[p1 - 1] + ' ' + mention
        if p2 >= len(tokens):
            mention = mention + ' <PAD>'
        else:
            mention = mention + ' ' + tokens[p2]

        offset = max(0, p1 - config.WINDOW_SIZE)
        text = ' '.join(tokens[offset:min(len(tokens), p2 + config.WINDOW_SIZE - 1)])
        p1 -= offset
        p2 -= offset

        out_type = []
        for a in types:
            flag = True
            for b in types:
                if len(a) >= len(b):
                    continue
                if (a == b[:len(a)]) and (b[len(a)] == '/'):
                    flag = False
            if flag:
                out_type.append(a)

        if full_path:
            outfile.write('%d\t%d\t%s\t%s\t%s\n' % (p1, p2, text, mention, ' '.join(types)))
        else:
            outfile.write('%d\t%d\t%s\t%s\t%s\n' % (p1, p2, text, mention, ' '.join(out_type)))
    outfile.close()


# noinspection PyShadowingNames
def parse_args(parser):
    parser.add_option('-d', '--data_name', type='string', dest='data_name')
    parser.add_option('-c', default=False, action='store_true', dest='if_clean')
    parser.add_option('-f', default=False, action='store_true', dest='full_path')

    (options, args) = parser.parse_args()
    return options, args


# noinspection PyShadowingNames
def main(options):
    preprocess(options.if_clean, options.full_path)


if __name__ == '__main__':
    o_parser = OptionParser()
    options, args = parse_args(o_parser)
    main(options)

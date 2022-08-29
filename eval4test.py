import os
import random
from optparse import OptionParser

from task4test_glc4 import Task
from utils import logging_utils
from model_param_space import param_space_dict
import datetime
import config
import tensorflow as tf


def parse_args(parser):
    parser.add_option('-m', '--model', dest='model_name', type='string')
    parser.add_option('-d', '--data', dest='data_name', type='string')
    parser.add_option('-p', '--portion', dest='portion', type=int, default=100)
    parser.add_option('-a', '--alpha', dest='alpha', type=float, default=0.)

    # ablation
    parser.add_option('-i', '--reinit', dest='re_init', action='store_true', default=False)
    parser.add_option('-g', '--gt_portion', dest='gt_portion', type='float', default=1.0)


    # sub exp
    parser.add_option('--exp4', dest='exp4', action='store_true', default=False)
    parser.add_option('--pseudo_label', dest='pseudo_label', type='float', default=1.0)

    parser.add_option('-o', '--savename', dest='save_name', type='string', default='')
    parser.add_option('-r', '--runs', dest='runs', type='int', default=5)
    parser.add_option('-s', '--single_run', dest='single', action='store_true', default=False)
    options, args = parser.parse_args()
    return options, args


def try_h_params():
    #以下是
    beta = [0.5]
    alpha = [0.2, 0]
    e_1 = [20]
    e_2 = [60]
    re_init = [False]
    aux_clusters = range(50)
    names = ['beta', 'alpha', 'e_1', 'e_2', 're_init', 'aux_clusters']
    res = []
    for ob in beta:
        for oa in alpha:
            for oe1 in e_1:
                for oe2 in e_2:
                    for orei in re_init:
                        for aux in aux_clusters:
                            c = [ob, oa, oe1, oe2, orei, aux]
                            res.append([(names[i], c[i]) for i in range(len(names))])
    # random.shuffle(res)
    return res


def update_h_params(h_params, options, one_try, logger):
    for one_param, value in one_try:
        if one_param in h_params:
            h_params[one_param] = value
            print(f'set {one_param} to {value}.')
            logger.info(f'set {one_param} to {value}.')
        if hasattr(options, one_param):
            setattr(options, one_param, value)
            print(f'set {one_param} to {value}.')
            logger.info(f'set {one_param} to {value}.')


def main(options):
    time_str = datetime.datetime.now().isoformat()
    if len(options.save_name) == 0:
        log_name = 'Eval_[Model@%s]_[Data@%s]_%s.log' % (options.model_name,
                                                         options.data_name, time_str)
    else:
        log_name = 'Eval_[Model@%s]_[Data@%s]_%s.log' % (options.save_name,
                                                         options.data_name, time_str)
    if os.name == 'nt':
        log_name = log_name.replace(':', '：')
    logger = logging_utils.get_logger(config.LOG_DIR, log_name)
    params_dict = param_space_dict[options.model_name]

    if options.single:
        print('Single run...')
        logger.info('Single run...')
        params_dict['alpha'] = options.alpha
        params_dict['gt_portion'] = options.gt_portion
        logger.info(f'hyper params: {params_dict}\noptions: {options}')
        print(f'hyper params: {params_dict}\noptions: {options}')
        params_dict['num_epochs'] = max(params_dict['e_1'] + params_dict['e_2'] + params_dict['e_3'],
                                        params_dict['num_epochs'])
        params_dict['pseudo_label'] = options.pseudo_label
        params_dict['exp4'] = options.exp4
        print(options.data_name)
        task = Task(model_name=options.model_name, data_name=options.data_name, cv_runs=options.runs,
                    params_dict=params_dict, logger=logger, portion=options.portion,
                    save_name=options.save_name, re_init=options.re_init)

        print('-' * 50 + 'refit' + '-' * 50)
        task.refit()
    else:
        for one_try in try_h_params():
            update_h_params(params_dict, options, one_try, logger)
            logger.info(f'hyper params: {params_dict}\noptions: {options}')
            print(f'hyper params: {params_dict}\noptions: {options}')
            params_dict['num_epochs'] = max(params_dict['e_1'] + params_dict['e_2'] + params_dict['e_3'],
                                            params_dict['num_epochs'])
            params_dict['pseudo_label'] = options.pseudo_label
            params_dict['exp4'] = options.exp4
            print(options.data_name)
            task = Task(model_name=options.model_name, data_name=options.data_name, cv_runs=options.runs,
                        params_dict=params_dict, logger=logger, portion=options.portion,
                        save_name=options.save_name, re_init=options.re_init)

            print('-' * 50 + 'refit' + '-' * 50)
            task.refit()
            del task
            tf.reset_default_graph()


if __name__ == '__main__':
    t_parser = OptionParser()
    opt, _ = parse_args(t_parser)
    main(opt)

from optparse import OptionParser

from task import Task
from utils import logging_utils
from model_param_space import param_space_dict
import datetime
import config


def parse_args(parser):
    parser.add_option('-m', '--model', dest='model_name', type='string')
    parser.add_option('-d', '--data', dest='data_name', type='string')
    parser.add_option('-p', '--portion', dest='portion', type=int, default=100)
    parser.add_option('-a', '--alpha', dest='alpha', type=float, default=0.)
    parser.add_option('-o', '--savename', dest='save_name', type='string', default='')
    parser.add_option('-r', '--runs', dest='runs', type='int', default=5)
    options, args = parser.parse_args()
    return options, args


def main(options):
    time_str = datetime.datetime.now().isoformat()
    time_str = time_str.replace(':', '-')
    if len(options.save_name) == 0:
        log_name = 'Eval_[Model@%s]_[Data@%s]_%s.log' % (options.model_name,
                                                         options.data_name, time_str)
    else:
        log_name = 'Eval_[Model@%s]_[Data@%s]_%s.log' % (options.save_name,
                                                         options.data_name, time_str)
    logger = logging_utils.get_logger(config.LOG_DIR, log_name)
    params_dict = param_space_dict[options.model_name]
    params_dict['num_epochs'] = max(params_dict['e_1'] + params_dict['e_2'] + params_dict['e_3'],
                                    params_dict['num_epochs'])
    task = Task(model_name=options.model_name, data_name=options.data_name, cv_runs=options.runs,
                params_dict=params_dict, logger=logger, portion=options.portion,
                save_name=options.save_name)

    print('-' * 50 + 'refit' + '-' * 50)
    task.refit()


if __name__ == '__main__':
    t_parser = OptionParser()
    opt, _ = parse_args(t_parser)
    main(opt)

"""
Author: ANTenna on 2021/12/25 5:09 下午
aliuyaohua@gmail.com

Description:

"""

import logging


def get_seg_line(val, seg_marker='-', seg_len=10):
    seg_line = seg_marker * seg_len
    return '{}  {}  {}'.format(seg_line, val, seg_line)


def create_logger(log_file_path='default_name.log', log_level='INFO', write_mode='w'):
    # create logger
    logger = logging.getLogger('logger')

    if log_level == 'INFO':
        logger.setLevel(logging.INFO)
    elif log_level == 'DEBUG':
        logger.setLevel(logging.DEBUG)
    else:
        raise KeyError('ERROR LOG LEVEL: {}'.format(log_level))

    # to log file
    file_handler = logging.FileHandler(log_file_path, write_mode)
    file_handler.setLevel(log_level)

    # to terminal
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(log_level)

    # set output format
    formatter = logging.Formatter("[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s")

    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


if __name__ == '__main__':
    print('Logger test ... ')

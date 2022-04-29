import logging
import time
import os


class Logger:
    def __init__(self, title="", verbose=True):
        # create logger
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)  # setting level
        formatter = logging.Formatter("[%(asctime)s] %(message)s")

        # create file handler
        start_time = time.strftime('%y-%m-%d-%H%M', time.localtime(time.time()))
        log_path = os.path.join(os.getcwd(), 'logs', title)  # setting path for logfile 设置日志文件名称
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        log_name = os.path.join(log_path, start_time + '.log')
        fh = logging.FileHandler(log_name, mode='w')
        fh.setLevel(logging.INFO)   # setting level for outputs in logfile
        fh.setFormatter(formatter)  # define format
        self.logger.addHandler(fh)  # add handler to the logger

        if verbose:
            # create console handler
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

    def info(self, msg):
        self.logger.info(msg)

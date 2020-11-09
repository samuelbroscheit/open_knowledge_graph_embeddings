import os
import logging.config
import json
import pandas

import pandas as pd

def setup_logging(log_file='log.txt', resume=False):
    """
    Setup logging configuration
    """
    if os.path.isfile(log_file) and resume:
        file_mode = 'a'
    else:
        file_mode = 'w'

    logger = logging.getLogger()
    if logger.hasHandlers():
        for handler in logger.handlers:
            logger.removeHandler(handler)
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s - %(levelname)s - %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                        filename=log_file,
                        filemode=file_mode)
    hasStreamHandler = False
    for handler in logger.handlers:
        if handler.__class__.__name__ == 'StreamHandler': hasStreamHandler = True
    if not hasStreamHandler:
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')
        console.setFormatter(formatter)
        logger.addHandler(console)
    return logger



class ResultsLog(object):

    supported_data_formats = ['csv', 'json']

    def __init__(self, path, resume=True, data_format='csv'):
        if data_format not in ResultsLog.supported_data_formats:
            raise ValueError('data_format must of the following: ' +
                             '|'.join(['{}'.format(k) for k in ResultsLog.supported_data_formats]))
        self.data_format = data_format
        os.makedirs(path, exist_ok=True)
        self.date_path = None
        self.set_path(path)
        self.results = pd.DataFrame()
        if os.path.isfile(self.date_path):
            if resume:
                self.load(self.date_path)

    def set_path(self, path):
        full_path = os.path.join(path, 'results')
        if self.data_format == 'json':
            self.date_path = '{}.json'.format(full_path)
        else:
            self.date_path = '{}.csv'.format(full_path)

    def clear(self):
        pass

    def add(self, **kwargs):
        """Add a new row to the dataframe
        example:
            resultsLog.add(epoch=epoch_num, train_loss=loss, test_loss=test_loss)
        """
        df = pd.DataFrame([kwargs.values()], columns=kwargs.keys())
        self.results = self.results.append(df, ignore_index=True, sort=False)

    def save(self, title='Training Results'):
        """save the json file.
        Parameters
        ----------
        title: string
            title of the HTML file
        """

        if self.data_format == 'json':
            self.results.to_json(self.date_path, orient='records', lines=True)
        else:
            self.results.to_csv(self.date_path, index=False, index_label=False)

    def load(self, path=None):
        """load the data file
        Parameters
        ----------
        path:
            path to load the json|csv file from
        """
        path = path or self.path
        if os.path.isfile(path):
            if self.data_format == 'json':
                self.results = pandas.read_json(path)
            else:
                self.results = pandas.read_csv(path)
        else:
            raise ValueError('{} isn''t a file'.format(path))

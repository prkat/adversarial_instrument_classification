import os
import csv


class Logger:
    def __init__(self, log_file_path, columns=None):
        if '~' in log_file_path:
            log_file_path = os.path.expanduser(log_file_path)
        self.log_path = log_file_path
        if columns is None:
            self.columns = ['epoch', 'train loss', 'train accuracy']
        else:
            self.columns = columns

        with open(self.log_path, 'w') as fp:
            writer = csv.writer(fp, delimiter=',')
            writer.writerow(columns)

    def append(self, value_list):
        with open(self.log_path, 'a') as fp:
            writer = csv.writer(fp, delimiter=',')
            writer.writerow(value_list)


import pandas as pd
import os
import random

cwd = os.getcwd()

def Dataloader(filepath, data_cols, num_rows, starting_rows, random_seed=False):
    # train_cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed']
    # test_cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time']
    dtypes = {
            'ip'            : 'uint32',
            'app'           : 'uint16',
            'device'        : 'uint16',
            'os'            : 'uint16',
            'channel'       : 'uint16',
            'is_attributed' : 'uint8',
            'click_id'      : 'uint32'
            }
    if random_seed:
        n = sum(1 for line in open(filepath)) - 1  # number of records in file (excludes header)
        # print(n)
        s = num_rows  # desired sample size
        skip = sorted(random.sample(range(1, n + 1), n - s))  # the 0-indexed header will not be included in the skip list
        # print(skip)
    else:
        print(starting_rows)
        skip = range(1, starting_rows)
    print(data_cols)
    data = pd.read_csv(filepath, skiprows=skip, nrows=num_rows, dtype=dtypes, usecols=data_cols)
    len_train = len(data)
    print('The initial size of the train set is', len_train)
    return data
import pandas as pd
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import gc
from features import Extra_Feature as extra_feature
from dataloader import Dataloader as dataloader
from trainer import train_lgb
import os
import argparse
from datetime import datetime


cwd = os.getcwd()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to run TD-lgb models')
    parser.add_argument('--not_debug', help='exit from debug mode', action='store_false', dest='debug', default=False)
    # parser.add_argument('--learning_rate', help='starting learning rate', action='store', type=float, dest='learning_rate', default=0.001)
    parser.add_argument('--load_model', help='load models from file', action='store_true', default=False)
    parser.add_argument('--predict', help='only predict', action='store_true', default=False)
    parser.add_argument('--validate', help='Using validation or not', action='store_true', default=True)
    parser.add_argument('--output_file', help='Prediction file name', action='store', default='results/submit_'+datetime.now().strftime('%m-%d_%H-%M')+'.csv')

    args = parser.parse_args()



    train_path = 'data/train/train_sample.csv'
    test_path = 'data/test/test.csv'
    train_cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed']
    # test_cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time']

    # train = dataloader(train_path, train_cols, 200)
    train = dataloader(filepath=train_path,
                       data_cols=train_cols,
                       num_rows=200,
                       starting_rows=0,
                       random_seed=True)
    # print("Train info before: \n\n")
    # print(train.info())
    train = extra_feature(train)
    gc.collect()
    # print("Train info after: \n\n")
    # print(train.info())

    # Train the model
    model = train_lgb(train=train,
                      validate=args.validate,
                      MAX_ROUNDS=1000,
                      EARLY_STOP=50,
                      OPT_ROUNDS=1000)

    predictors = ['app','device',  'os', 'channel', 'hour', 'n_channels', 'ip_app_count', 'ip_app_os_count',
                 'ip_tchan_count','ip_app_os_var','ip_app_channel_var_day','ip_app_channel_mean_hour','nip_hh_dev']

    # Prediction
    VALID_OUTFILE = 'sub_lgbm_v.csv'
    outfile = args.output_file


    print('load test...')
    test_cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'click_id']
    # test_df = pd.read_csv(test_path, dtype=dtypes, usecols=test_cols)
    # test_df = dataloader(test_path, test_cols, 20)
    test_batch_size = 100
    nrows_in_total = sum(1 for line in open(test_path)) - 1  # number of records in file (excludes header)
    nbatch = int((nrows_in_total-1)/test_batch_size)+1
    nbatch = 3
    for count in range(0, nbatch):
        print(count)
        starting_pos = 1+count*test_batch_size
        # num_of_rows = test_batch_size if count is not nbatch-1 else nrows_in_total-(nbatch-1)*test_batch_size
        num_of_rows = test_batch_size if count is not nbatch - 1 else 20
        print(num_of_rows, starting_pos)
        test_df = dataloader(filepath=test_path,
                             data_cols=test_cols,
                             num_rows=num_of_rows,
                             starting_rows=starting_pos,
                             random_seed=False)

        test_df = extra_feature(test_df)
        gc.collect()

        sub = pd.DataFrame()
        sub['click_id'] = test_df['click_id']

        print("Predicting...")
        sub['is_attributed'] = model.predict(test_df[predictors])
        print("writing...")
        with open(outfile, 'a') as f:
            if count is 0:
                sub.to_csv(f, index=False, float_format='%.9f')
            else:
                sub.to_csv(f, index=False, header=False, float_format='%.9f')
        print('pp')
        del test_df
    print("done...")
    print(sub.info())



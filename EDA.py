import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load subset of the training data
X_train = pd.read_csv('data/train/train_sample.csv', nrows=20, parse_dates=['click_time'])

# Show the head of the table
# print(X_train.head())

X_train['day'] = X_train['click_time'].dt.day.astype('uint8')
X_train['hour'] = X_train['click_time'].dt.hour.astype('uint8')
X_train['minute'] = X_train['click_time'].dt.minute.astype('uint8')
X_train['second'] = X_train['click_time'].dt.second.astype('uint8')
# print(X_train.head())

ATTRIBUTION_CATEGORIES = [
    # V1 Features #
    ###############
    ['ip'], ['app'], ['device'], ['os'], ['channel'],

    # V2 Features #
    ###############
    ['app', 'channel'],
    ['app', 'os'],
    ['app', 'device'],

    # V3 Features #
    ###############
    ['channel', 'os'],
    ['channel', 'device'],
    ['os', 'device']
]
new_attribution = [['os']]

# Find frequency of is_attributed for each unique value in column
freqs = {}
for cols in ATTRIBUTION_CATEGORIES:
    # New feature name
    new_feature = '_'.join(cols) + '_confRate'

    # Perform the groupby
    group_object = X_train.groupby(cols)
    # print('*******')
    # print(group_object)
    # group_object.boxplot()
    # plt.show()
    # print('*******')
    # Group sizes
    group_sizes = group_object.size()
    # print(group_sizes)
    log_group = np.log(100000)  # 1000 views -> 60% confidence, 100 views -> 40% confidence
    print(
        ">> Calculating confidence-weighted rate for: {}.\n   Saving to: {}. Group Max /Mean / Median / Min: {} / {} / {} / {}".format(
            cols, new_feature,
            group_sizes.max(),
            np.round(group_sizes.mean(), 2),
            np.round(group_sizes.median(), 2),
            group_sizes.min()
        ))


    # Aggregation function
    def rate_calculation(x):
        # print('----------')
        # print(x)
        # print('----------')
        # Calculate the attributed rate. Scale by confidence
        rate = x.sum() / float(x.count())
        conf = np.min([1, np.log(x.count()) / log_group])
        return rate * conf


    # Perform the merge
    X_train = X_train.merge(
        group_object['is_attributed']. \
            apply(rate_calculation). \
            reset_index(). \
            rename(
            index=str,
            columns={'is_attributed': new_feature}
        )[cols + [new_feature]],
        on=cols, how='left'
    )

X_train.head()
print(X_train)

GROUPBY_AGGREGATIONS = [

    # V1 - GroupBy Features #
    #########################
    # Variance in day, for ip-app-channel
    {'groupby': ['ip', 'app', 'channel'], 'select': 'day', 'agg': 'var'},
    # Variance in hour, for ip-app-os
    {'groupby': ['ip', 'app', 'os'], 'select': 'hour', 'agg': 'var'},
    # Variance in hour, for ip-day-channel
    {'groupby': ['ip', 'day', 'channel'], 'select': 'hour', 'agg': 'var'},
    # Count, for ip-day-hour
    {'groupby': ['ip', 'day', 'hour'], 'select': 'channel', 'agg': 'count'},
    # Count, for ip-app
    {'groupby': ['ip', 'app'], 'select': 'channel', 'agg': 'count'},
    # Count, for ip-app-os
    {'groupby': ['ip', 'app', 'os'], 'select': 'channel', 'agg': 'count'},
    # Count, for ip-app-day-hour
    {'groupby': ['ip', 'app', 'day', 'hour'], 'select': 'channel', 'agg': 'count'},
    # Mean hour, for ip-app-channel
    {'groupby': ['ip', 'app', 'channel'], 'select': 'hour', 'agg': 'mean'},

    # V2 - GroupBy Features #
    #########################
    # Average clicks on app by distinct users; is it an app they return to?
    {'groupby': ['app'],
     'select': 'ip',
     'agg': lambda x: float(len(x)) / len(x.unique()),
     'agg_name': 'AvgViewPerDistinct'
     },
    # How popular is the app or channel?
    {'groupby': ['app'], 'select': 'channel', 'agg': 'count'},
    {'groupby': ['channel'], 'select': 'app', 'agg': 'count'},

    # V3 - GroupBy Features                                              #
    # https://www.kaggle.com/bk0000/non-blending-lightgbm-model-lb-0-977 #
    ######################################################################
    {'groupby': ['ip'], 'select': 'channel', 'agg': 'nunique'},
    {'groupby': ['ip'], 'select': 'app', 'agg': 'nunique'},
    {'groupby': ['ip', 'day'], 'select': 'hour', 'agg': 'nunique'},
    {'groupby': ['ip', 'app'], 'select': 'os', 'agg': 'nunique'},
    {'groupby': ['ip'], 'select': 'device', 'agg': 'nunique'},
    {'groupby': ['app'], 'select': 'channel', 'agg': 'nunique'},
    {'groupby': ['ip', 'device', 'os'], 'select': 'app', 'agg': 'nunique'},
    {'groupby': ['ip', 'device', 'os'], 'select': 'app', 'agg': 'cumcount'},
    {'groupby': ['ip'], 'select': 'app', 'agg': 'cumcount'},
    {'groupby': ['ip'], 'select': 'os', 'agg': 'cumcount'},
    {'groupby': ['ip', 'day', 'channel'], 'select': 'hour', 'agg': 'var'}
]
new_att = [
    {'groupby': ['ip', 'app', 'channel'], 'select': 'day', 'agg': 'var'}
]
# Apply all the groupby transformations
for spec in new_att:

    # Name of the aggregation we're applying
    agg_name = spec['agg_name'] if 'agg_name' in spec else spec['agg']

    # Name of new feature
    new_feature = '{}_{}_{}'.format('_'.join(spec['groupby']), agg_name, spec['select'])

    # Info
    print("Grouping by {}, and aggregating {} with {}".format(
        spec['groupby'], spec['select'], agg_name
    ))

    # Unique list of features to select
    all_features = list(set(spec['groupby'] + [spec['select']]))

    # Perform the groupby
    gp = X_train[all_features]. \
        groupby(spec['groupby'])[spec['select']]. \
        agg(spec['agg']). \
        reset_index(). \
        rename(index=str, columns={spec['select']: new_feature})

    # Merge back to X_total
    if 'cumcount' == spec['agg']:
        X_train[new_feature] = gp[0].values
    else:
        X_train = X_train.merge(gp, on=spec['groupby'], how='left')

    # Clear memory
    del gp
    gc.collect()

print(X_train)
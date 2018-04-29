from sklearn.model_selection import train_test_split
import lightgbm as lgb
import gc


def train_lgb(train, validate, MAX_ROUNDS, EARLY_STOP, OPT_ROUNDS):
    metrics = 'auc'
    lgb_params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': metrics,
        'learning_rate': 0.05,
        'num_leaves': 31,  # we should let it be smaller than 2^(max_depth)
        'max_depth': -1,  # -1 means no limit
        'min_child_samples': 20,  # Minimum number of data need in a child(min_data_in_leaf)
        'max_bin': 255,  # Number of bucketed bin for feature values
        'subsample': 0.7,  # Subsample ratio of the training instance.
        'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
        'colsample_bytree': 0.7,  # Subsample ratio of columns when constructing each tree.
        'subsample_for_bin': 200000,  # Number of samples for constructing bin
        'min_child_weight': 5,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
        'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
        'nthread': 8,
        'verbose': 0,
        'scale_pos_weight': 99.7,  # because training data is extremely unbalanced
        'metric': metrics
    }

    target = 'is_attributed'
    train[target] = train[target].astype('uint8')

    predictors = ['app', 'device', 'os', 'channel', 'hour', 'n_channels', 'ip_app_count', 'ip_app_os_count',
                  'ip_tchan_count', 'ip_app_os_var', 'ip_app_channel_var_day', 'ip_app_channel_mean_hour', 'nip_hh_dev']
    categorical = ['app', 'device', 'os', 'channel']
    gc.collect()

    MAX_ROUNDS = 1000
    EARLY_STOP = 50
    OPT_ROUNDS = 1000


    if validate:
        train, val_df = train_test_split(train, train_size=.95, shuffle=False)

        print("\nTrain Information ", train.info())
        print("\nVal Information ", val_df.info())

        print("train size: ", len(train))
        print("valid size: ", len(val_df))
        gc.collect()

        print("Training...\n")

        num_boost_round = MAX_ROUNDS
        early_stopping_rounds = EARLY_STOP

        xgtrain = lgb.Dataset(train[predictors].values, label=train[target].values,
                              feature_name=predictors,
                              categorical_feature=categorical
                              )
        del train
        gc.collect()

        xgvalid = lgb.Dataset(val_df[predictors].values, label=val_df[target].values,
                              feature_name=predictors,
                              categorical_feature=categorical
                              )
        del val_df
        gc.collect()

        evals_results = {}

        bst = lgb.train(lgb_params,
                        xgtrain,
                        valid_sets=[xgvalid],
                        valid_names=['valid'],
                        evals_result=evals_results,
                        num_boost_round=num_boost_round,
                        early_stopping_rounds=early_stopping_rounds,
                        verbose_eval=10,
                        feval=None)

        n_estimators = bst.best_iteration

        print("\nModel Report")
        print("n_estimators : ", n_estimators)
        print(metrics + ":", evals_results['valid'][metrics][n_estimators - 1])


        del xgvalid

    else:

        print("train size: ", len(train))

        gc.collect()

        print("Training...")

        num_boost_round = OPT_ROUNDS

        xgtrain = lgb.Dataset(train[predictors].values, label=train[target].values,
                              feature_name=predictors,
                              categorical_feature=categorical
                              )
        del train
        gc.collect()
        print("Dataset preparing done")

        bst = lgb.train(lgb_params,
                        xgtrain,
                        num_boost_round=num_boost_round,
                        verbose_eval=10,
                        feval=None)
        print("Traing done")

    del xgtrain
    gc.collect()
    return bst

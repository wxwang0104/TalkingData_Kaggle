import pandas as pd
import gc


def Extra_Feature(df):
    print("Creating new time features: 'hour' and 'day'...")
    df['hour'] = pd.to_datetime(df.click_time).dt.hour.astype('uint8')
    df['day'] = pd.to_datetime(df.click_time).dt.day.astype('uint8')
    gc.collect()

    print("Feature Engineering \n")

    print('1. Computing the number of clicks associated with a given IP address within each hour... ')
    n_channel = df[['ip', 'day', 'hour', 'channel']].groupby(by=['ip', 'day',
                                                                 'hour'])[['channel']].count().reset_index().rename(
        columns={'channel': 'n_channels'})
    print('Merging the channels data with the main data set...\n')
    df = df.merge(n_channel, on=['ip', 'day', 'hour'], how='left')
    del n_channel
    gc.collect()

    print('2. Computing the number of clicks associated with a given IP address and app...')
    n_channel = df[['ip', 'app', 'channel']].groupby(by=['ip',
                                                         'app'])[['channel']].count().reset_index().rename(
        columns={'channel': 'ip_app_count'})
    print('Merging the channels data with the main data set...\n')
    df = df.merge(n_channel, on=['ip', 'app'], how='left')
    del n_channel
    gc.collect()

    print('3. Computing the number of clicks associated with a given IP address, app, and os...')
    n_channel = df[['ip', 'app', 'os', 'channel']].groupby(by=['ip', 'app',
                                                               'os'])[['channel']].count().reset_index().rename(
        columns={'channel': 'ip_app_os_count'})
    print('Merging the channels data with the main data set...\n')
    df = df.merge(n_channel, on=['ip', 'app', 'os'], how='left')
    del n_channel
    gc.collect()

    # Adding features with var and mean hour (inspired from nuhsikander's script)

    print('4. grouping by : ip_day_chl_var_hour..')
    n_channel = df[['ip', 'day', 'hour', 'channel']].groupby(by=['ip', 'day',
                                                                 'channel'])[['hour']].var().reset_index().rename(
        columns={'hour': 'ip_tchan_count'})
    print('Merging the hour data with the main data set...\n')
    df = df.merge(n_channel, on=['ip', 'day', 'channel'], how='left')
    del n_channel
    gc.collect()

    print('5. grouping by : ip_app_os_var_hour..')
    n_channel = df[['ip', 'app', 'os', 'hour']].groupby(by=['ip', 'app',
                                                            'os'])[['hour']].var().reset_index().rename(index=str,
                                                                                                        columns={
                                                                                                            'hour': 'ip_app_os_var'})
    print('Merging the hour data with the main data set...\n')
    df = df.merge(n_channel, on=['ip', 'app', 'os'], how='left')
    del n_channel
    gc.collect()

    print('6. grouping by : ip_app_channel_var_day...')
    n_channel = df[['ip', 'app', 'channel', 'day']].groupby(by=['ip', 'app',
                                                                'channel'])[['day']].var().reset_index().rename(
        columns={'day': 'ip_app_channel_var_day'})
    print('Merging the day data with the main data set...\n')
    df = df.merge(n_channel, on=['ip', 'app', 'channel'], how='left')
    del n_channel
    gc.collect()

    print('7. grouping by : ip_app_chl_mean_hour..')
    n_channel = df[['ip', 'app', 'channel', 'hour']].groupby(by=['ip', 'app',
                                                                 'channel'])[['hour']].mean().reset_index().rename(
        columns={'hour': 'ip_app_channel_mean_hour'})
    print('Merging the meanhour data with the main data set...\n')
    df = df.merge(n_channel, on=['ip', 'app', 'channel'], how='left')
    del n_channel
    gc.collect()

    print('8. group by : ip_hh_dev')
    n_channel = df[['ip', 'device', 'hour', 'day', 'channel']].groupby(by=['ip', 'device', 'day',
                                                                           'hour'])[
        ['channel']].count().reset_index().rename(index=str, columns={'channel': 'nip_hh_dev'})
    print('Merging the channel data with the main data set...\n')
    df = df.merge(n_channel, on=['ip', 'device', 'day', 'hour'], how='left')
    del n_channel
    gc.collect()

    print('9. debug')
    n_channel = df[['ip']]
    # print(df['ip_tchan_count'])

    df['n_channels'] = df['n_channels'].astype('uint16')
    df['ip_app_count'] = df['ip_app_count'].astype('uint16')
    df['ip_app_os_count'] = df['ip_app_os_count'].astype('uint16')
    df['ip_tchan_count'] = df['ip_tchan_count'].astype('float32')
    df['ip_app_os_var'] = df['ip_app_os_var'].astype('float32')
    df['ip_app_channel_var_day'] = df['ip_app_channel_var_day'].astype('float32')
    df['ip_app_channel_mean_hour'] = df['ip_app_channel_mean_hour'].astype('float32')
    df['nip_hh_dev'] = df['nip_hh_dev'].astype('uint16')


    df.drop(['ip', 'day'], axis=1, inplace=True)
    gc.collect()
    return df
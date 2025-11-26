import numpy as np
import pandas as pd
import datetime
from sklearn.model_selection import train_test_split


def process_data(filename='commuting_data.csv'):
    '''Import the data from a csv file and perform initial processing tasks.

    Args:
        filename (csv file, optional): The csv file with the data.

    Returns
        pandas DataFrame
    '''

    df = pd.read_csv(filename)

    if 'ferry' in filename:
        time_columns = ['home_departure_time',
                        'work_departure_time',
                        'work_arrival_time',
                        'home_arrival_time',
                        'park_in_line_southworth',
                        'park_on_southworth_ferry_time',
                        'southworth_ferry_launch_time',
                        'fauntleroy_ferry_departure_time']
    else:
        time_columns = ['home_departure_time',
                        'work_departure_time',
                        'work_arrival_time',
                        'home_arrival_time']

    # store times as datetime types
    for ts in time_columns:
        if ts in df.columns:
            df[ts] = pd.to_datetime(df['date'] + ' ' + df[ts], errors='coerce')

    # Calculate minutes after midnight for departure and arrival times
    # and store these as new columns
    midnight = datetime.time()

    # Adding columns: calculate commuting minutes;
    # remove the date info from departure time, preserving just the time info;
    # calculate the mileage
    if 'work_arrival_time' in df.columns:
        df['minutes_to_work'] = (
            df['work_arrival_time'] - df['home_departure_time']
        ).dt.total_seconds()/60
        df['home_departure_time_hr'] = (
            df['home_departure_time'] -
            pd.to_datetime(df['date'] + ' ' + str(midnight))
        ).dt.total_seconds()/(60*60)
        df['mileage_to_work'] = (df['work_arrival_mileage'] -
                                 df['home_departure_mileage'])
        df['work_arrival_time_hr'] = (
            df['work_arrival_time'] -
            pd.to_datetime(df['date'] + ' ' + str(midnight))
        ).dt.total_seconds()/(60*60)
        # Subtract 4 minutes from 'minutes_to_work' if 'gas' appears
        # in that row under the column 'comments_from_home_to_work'
        if ('comments_from_home_to_work' in df.columns and
                df['comments_from_home_to_work'].notna().any()):
            df.loc[df['comments_from_home_to_work'].str.contains(
                'gas', na=False), 'minutes_to_work'] -= 4
    if 'home_arrival_time' in df.columns:
        df['minutes_to_home'] = (
            df['home_arrival_time'] - df['work_departure_time']
        ).dt.total_seconds()/60
        df['work_departure_time_hr'] = (
            df['work_departure_time'] -
            pd.to_datetime(df['date'] + ' ' + str(midnight))
        ).dt.total_seconds()/(60*60)
        df['mileage_to_home'] = (df['home_arrival_mileage'] -
                                 df['work_departure_mileage'])
        df['home_arrival_time_hr'] = (
            df['home_arrival_time'] -
            pd.to_datetime(df['date'] + ' ' + str(midnight))
        ).dt.total_seconds()/(60*60)
        # Subtract 4 minutes from 'minutes_to_home' if 'gas' appears
        # in that row under the column 'comments_from_work_to_home'
        if ('comments_from_work_to_home' in df.columns and
                df['comments_from_work_to_home'].notna().any()):
            df.loc[df['comments_from_work_to_home'].str.contains(
                'gas', na=False), 'minutes_to_home'] -= 4

    # Adding column for month to help explore seasonality in the data
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter

    return df


def preprocess_data(start, end, df):
    # print(df)
    df = pd.get_dummies(data=df, columns=['day_of_week'], drop_first=True)

    # Cyclical encoding of month for seasonality
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    df_notna = df[df[start + '_departure_time_hr'].notna()]
    features = [start + '_departure_time_hr']
    weekday_columns = []
    for col in df_notna.columns:
        if col[:11] == 'day_of_week':
            weekday_columns.append(col)
    features.extend(weekday_columns)
    features.extend(['quarter', 'month_sin', 'month_cos'])

    # Filter out outliers
    print(f'min: {int(df_notna["minutes_to_" + end].min())}')
    print(f'max: {int(df_notna["minutes_to_" + end].max())}')
    mean = df_notna["minutes_to_" + end].mean()
    std = df_notna["minutes_to_" + end].std()
    print(f'mean: {round(mean, 2)}')
    print(f'3 * std: {round(3 * std, 2)}')
    print(f'mean + 3 * std: {round(mean + 3 * std, 2)}')
    df_ready = df_notna[
        np.abs(df_notna["minutes_to_" + end] - mean) <= 3 * std
    ]
    df_filtered_out = df_notna[
        ~(np.abs(df_notna["minutes_to_" + end] - mean) <= 3 * std)
    ]
    print('Filtered out:')
    print(df_filtered_out[['date', 'minutes_to_' + end]])

    X = df_ready[features]
    y = np.array(df_ready['minutes_to_' + end])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                        random_state=13)
    return X_train, X_test, y_train, y_test

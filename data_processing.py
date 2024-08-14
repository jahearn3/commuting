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
        time_columns = ['home_departure_time', 'work_departure_time', 'work_arrival_time', 'home_arrival_time', 'park_in_line_southworth', 'park_on_southworth_ferry_time', 'southworth_ferry_launch_time', 'fauntleroy_ferry_departure_time']
    else:
        time_columns = ['home_departure_time', 'work_departure_time', 'work_arrival_time', 'home_arrival_time']

    # store times as datetime types
    for ts in time_columns:
        if ts in df.columns:
            df[ts] = pd.to_datetime(df['date'] + ' ' + df[ts], errors='coerce')

    # Calculate minutes after midnight for departure and arrival times and store these as new columns
    midnight = datetime.time()

    # Adding columns: calculate commuting minutes; remove the date info from departure time, preserving just the time info; calculate the mileage
    if 'work_arrival_time' in df.columns:
        start = 'home'
        end = 'work'
        df['minutes_to_work'] = (df['work_arrival_time'] - df['home_departure_time']).dt.total_seconds()/60
        df['home_departure_time_hr'] = (df['home_departure_time'] - pd.to_datetime(df['date'] + ' ' + str(midnight))).dt.total_seconds()/(60*60)
        df['mileage_to_work'] = df['work_arrival_mileage'] - df['home_departure_mileage']
        df['work_arrival_time_hr'] = (df['work_arrival_time'] - pd.to_datetime(df['date'] + ' ' + str(midnight))).dt.total_seconds()/(60*60)
    if 'home_arrival_time' in df.columns:
        start = 'work'
        end = 'home'
        df['minutes_to_home'] = (df['home_arrival_time'] - df['work_departure_time']).dt.total_seconds()/60
        df['work_departure_time_hr'] = (df['work_departure_time'] - pd.to_datetime(df['date'] + ' ' + str(midnight))).dt.total_seconds()/(60*60)
        df['mileage_to_home'] = df['home_arrival_mileage'] - df['work_departure_mileage']
        df['home_arrival_time_hr'] = (df['home_arrival_time'] - pd.to_datetime(df['date'] + ' ' + str(midnight))).dt.total_seconds()/(60*60)

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
    print(f'mean: {round(df_notna["minutes_to_" + end].mean(), 2)}')
    print(f'3 * std: {round(3 * df_notna["minutes_to_" + end].std(), 2)}')
    print(f'mean + 3 * std: {round(df_notna["minutes_to_" + end].mean() + 3 * df_notna["minutes_to_" + end].std(), 2)}')
    mean_time = df_notna["minutes_to_" + end].mean()
    df_ready = df_notna[np.abs(df_notna["minutes_to_" + end] - mean_time) <= (3 * df_notna["minutes_to_" + end].std())]
    df_filtered_out = df_notna[~(np.abs(df_notna["minutes_to_" + end] - mean_time) <= (3 * df_notna["minutes_to_" + end].std()))]
    print('Filtered out:')
    print(df_filtered_out[['date', 'minutes_to_' + end]])

    X = df_ready[features]
    # X = df_notna[[start + '_departure_time_hr','day_of_week_Mon','day_of_week_Tue','day_of_week_Wed','day_of_week_Thu']]
    # df = df.get_dummies(data=df, prefix=['Mon', 'Tue', 'Wed', 'Thu', 'Fri'], columns='day_of_week', drop_first=True)
    # X = np.array(df[start + '_departure_time_hr'].dropna()).reshape(-1, 1)
    # X = np.array(df_notna[start + '_departure_time_hr']).reshape(-1, 1)
    # X = np.array([df_notna[start + '_departure_time_hr'], df_notna['day_of_week_Mon'], df_notna['day_of_week_Tue'], df_notna['day_of_week_Wed'], df_notna['day_of_week_Thu']])
    # print(X.shape)
    # X = np.array([df[start + '_departure_time_hr'], df['day_of_week_Mon'], df['day_of_week_Tue'], df['day_of_week_Wed'], df['day_of_week_Thu']])
    # y = np.array(df['minutes_to_' + end].dropna())
    y = np.array(df_ready['minutes_to_' + end])
    # (len1, len2) = X.shape
    # X = np.reshape(X, (len2, len1))
    # print(y)
    # X = df_notna.drop(['minutes_to_' + end], axis=1)
    # print(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=13)
    return X_train, X_test, y_train, y_test 


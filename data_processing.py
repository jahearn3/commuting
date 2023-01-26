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

    # store times as datetime types
    df['home_departure_time'] = pd.to_datetime(df['date'] + ' ' + df['home_departure_time'], errors='coerce')
    df['work_departure_time'] = pd.to_datetime(df['date'] + ' ' + df['work_departure_time'], errors='coerce')
    df['work_arrival_time'] = pd.to_datetime(df['date'] + ' ' + df['work_arrival_time'], errors='coerce')
    df['home_arrival_time'] = pd.to_datetime(df['date'] + ' ' + df['home_arrival_time'], errors='coerce')

    # calculate commuting minutes and store these as new columns
    df['minutes_to_work'] = (df['work_arrival_time'] - df['home_departure_time']).dt.total_seconds()/60
    df['minutes_to_home'] = (df['home_arrival_time'] - df['work_departure_time']).dt.total_seconds()/60

    # calculate minutes after midnight for departure and arrival times and store these as new columns
    midnight = datetime.time()
    
    #print(df['home_departure_time'])

    #df['home_departure_time'] = datetime.datetime.strptime(df['home_departure_time'], '%H:%M')
    df['home_departure_time_hr'] = (df['home_departure_time'] - pd.to_datetime(df['date'] + ' ' + str(midnight))).dt.total_seconds()/(60*60)
    df['work_departure_time_hr'] = (df['work_departure_time'] - pd.to_datetime(df['date'] + ' ' + str(midnight))).dt.total_seconds()/(60*60)
    #print(df['home_departure_time_hr'])

    #print('Mean duration: {:.2f} minutes'.format(df['minutes_to_work'].mean()))
    #print('Median duration: {:.0f} minutes'.format(df['minutes_to_work'].median()))
    #print(df['home_departure_time'].dt.day_name())

    #print(type(pd.iloc()))
    df['mileage_to_work'] = df['work_arrival_mileage'] - df['home_departure_mileage']
    df['mileage_to_home'] = df['home_arrival_mileage'] - df['work_departure_mileage']

    return df

#df = process_data()

def preprocess_data(start, end, df):
    #print(df)
    df = pd.get_dummies(data=df, columns=['day_of_week'], drop_first=True)
    df_notna = df[df[start + '_departure_time_hr'].notna()]
    X = df_notna[[start + '_departure_time_hr','day_of_week_Mon','day_of_week_Tue','day_of_week_Wed','day_of_week_Thu']]
    #df = df.get_dummies(data=df, prefix=['Mon', 'Tue', 'Wed', 'Thu', 'Fri'], columns='day_of_week', drop_first=True)
    #X = np.array(df[start + '_departure_time_hr'].dropna()).reshape(-1, 1)
    #X = np.array(df_notna[start + '_departure_time_hr']).reshape(-1, 1)
    #X = np.array([df_notna[start + '_departure_time_hr'], df_notna['day_of_week_Mon'], df_notna['day_of_week_Tue'], df_notna['day_of_week_Wed'], df_notna['day_of_week_Thu']])
    #print(X.shape)
    #X = np.array([df[start + '_departure_time_hr'], df['day_of_week_Mon'], df['day_of_week_Tue'], df['day_of_week_Wed'], df['day_of_week_Thu']])
    #y = np.array(df['minutes_to_' + end].dropna())
    y = np.array(df_notna['minutes_to_' + end])
    #(len1, len2) = X.shape
    #X = np.reshape(X, (len2, len1))
    #print(y)
    #X = df_notna.drop(['minutes_to_' + end], axis=1)
    #print(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=13)
    return X_train, X_test, y_train, y_test 


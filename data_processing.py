import pandas as pd 
import datetime

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
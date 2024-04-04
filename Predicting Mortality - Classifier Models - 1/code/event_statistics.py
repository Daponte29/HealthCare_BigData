import time
import pandas as pd
import numpy as np


def read_csv(filepath):
    '''
    TODO : This function needs to be completed.
    Read the events.csv and mortality_events.csv files. 
    Variables returned from this function are passed as input to the metric functions.
    '''
    events = pd.read_csv(filepath + 'events.csv')
    mortality = pd.read_csv(filepath + 'mortality_events.csv')

    return events, mortality

def event_count_metrics(events, mortality):
    '''
    TODO : Implement this function to return the event count metrics.
    Event count is defined as the number of events recorded for a given patient.
    '''
    #Group Alive and Dead Patients
    merged_df = pd.merge(events, mortality, on='patient_id', how='left')
    
    dead_patients = merged_df[merged_df['patient_id'].isin(mortality['patient_id'])] #Grouped dead patients
    alive_patients = merged_df[~merged_df['patient_id'].isin(mortality['patient_id'])] #Grouped alive patients
    #Count events for each patient_id
    dead_count = dead_patients.groupby('patient_id').size().reset_index(name='count')
    alive_count = alive_patients.groupby('patient_id').size().reset_index(name='count')
    # u_dead=mortality['patient_id'].nunique()
    # u_dead2=dead_patients['patient_id'].nunique()
    
    avg_dead_event_count = dead_count['count'].mean()
    max_dead_event_count = dead_count['count'].max()
    min_dead_event_count = dead_count['count'].min()
    avg_alive_event_count = alive_count['count'].mean()
    max_alive_event_count = alive_count['count'].max()
    min_alive_event_count = alive_count['count'].min()

    return min_dead_event_count, max_dead_event_count, avg_dead_event_count, min_alive_event_count, max_alive_event_count, avg_alive_event_count

def encounter_count_metrics(events, mortality):
    '''
    TODO : Implement this function to return the encounter count metrics.
    Encounter count is defined as the count of unique dates on which a given patient visited the ICU. 
    '''
    #Group Alive and Dead Patients
    merged_df = pd.merge(events, mortality, on='patient_id', how='left')
    dead_patients = merged_df[merged_df['patient_id'].isin(mortality['patient_id'])] #Grouped dead patients
    alive_patients = merged_df[~merged_df['patient_id'].isin(mortality['patient_id'])] #Grouped alive patients
    # Convert timestamp to datetime data type
    #dead_patients['timestamp'] = pd.to_datetime(dead_patients['timestamp_x'])
    #alive_patients['timestamp'] = pd.to_datetime(alive_patients['timestamp_x'])

    # Count unique dates for dead patients
    dead_unique_dates = dead_patients.groupby('patient_id')['timestamp_x'].nunique().reset_index(name='unique_dates')
    # Count unique dates for alive patients
    alive_unique_dates = alive_patients.groupby('patient_id')['timestamp_x'].nunique().reset_index(name='unique_dates')

    avg_dead_encounter_count = dead_unique_dates['unique_dates'].mean()
    max_dead_encounter_count = dead_unique_dates['unique_dates'].max()
    min_dead_encounter_count = dead_unique_dates['unique_dates'].min() 
    avg_alive_encounter_count = alive_unique_dates['unique_dates'].mean()
    max_alive_encounter_count = alive_unique_dates['unique_dates'].max()
    min_alive_encounter_count = alive_unique_dates['unique_dates'].min()

    return min_dead_encounter_count, max_dead_encounter_count, avg_dead_encounter_count, min_alive_encounter_count, max_alive_encounter_count, avg_alive_encounter_count

def record_length_metrics(events, mortality):
    '''
    TODO: Implement this function to return the record length metrics.
    Record length is the duration between the first event and the last event for a given patient. 
    '''
    #Group Alive and Dead Patients
    merged_df = pd.merge(events, mortality, on='patient_id', how='left')

    dead_patients = merged_df[merged_df['patient_id'].isin(mortality['patient_id'])] #Grouped dead patients
    alive_patients = merged_df[~merged_df['patient_id'].isin(mortality['patient_id'])] #Grouped alive patients
    
    #dead_patients['timestamp_x'] = pd.to_datetime(dead_patients['timestamp_x'])
    #alive_patients['timestamp_x'] = pd.to_datetime(alive_patients['timestamp_x'])

    # Define a function to calculate the duration
    def calculate_duration(group):
        group = pd.to_datetime(group) # makee sure its datetime to do the aggregation operation in return statement for days
        if len(group) > 1:
            return (group.max() - group.min()).days
        else:
            return 0

    # Calculate duration for dead patients
    dead_record = dead_patients.groupby('patient_id')['timestamp_x'].apply(calculate_duration).reset_index(name='duration')
    #dead_record['duration'] = pd.to_timedelta(dead_record['duration'], unit='D')
    
    # Calculate duration for alive patients
    alive_record = alive_patients.groupby('patient_id')['timestamp_x'].apply(calculate_duration).reset_index(name='duration')
    #alive_record['duration'] = pd.to_timedelta(alive_record['duration'], unit='D')


    avg_dead_rec_len = dead_record['duration'].mean()
    max_dead_rec_len = dead_record['duration'].max()
    min_dead_rec_len = dead_record['duration'].min()
    avg_alive_rec_len = alive_record['duration'].mean()
    max_alive_rec_len = alive_record['duration'].max()
    min_alive_rec_len = alive_record['duration'].min()

    return min_dead_rec_len, max_dead_rec_len, avg_dead_rec_len, min_alive_rec_len, max_alive_rec_len, avg_alive_rec_len

def main():
  
   
    train_path = '../data/train/'

    events, mortality = read_csv(train_path)

    #Compute the event count metrics
    start_time = time.time()
    event_count = event_count_metrics(events, mortality)
    end_time = time.time()
    print(("Time to compute event count metrics: " + str(end_time - start_time) + "s"))
    print(event_count)

    #Compute the encounter count metrics
    start_time = time.time()
    encounter_count = encounter_count_metrics(events, mortality)
    end_time = time.time()
    print(("Time to compute encounter count metrics: " + str(end_time - start_time) + "s"))
    print(encounter_count)

    #Compute record length metrics
    start_time = time.time()
    record_length = record_length_metrics(events, mortality)
    end_time = time.time()
    print(("Time to compute record length metrics: " + str(end_time - start_time) + "s"))
    print(record_length)
    
if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 12:51:05 2024

@author: nolot
"""
import etl
import utils
import time
import pandas as pd
import numpy as np

# PLEASE USE THE GIVEN FUNCTION NAME, DO NOT CHANGE IT

def read_csv(filepath):
    
    '''
    TODO: This function needs to be completed.
    Read the events.csv, mortality_events.csv and event_feature_map.csv files into events, mortality and feature_map.
    
    Return events, mortality and feature_map
    '''

    #Columns in events.csv - patient_id,event_id,event_description,timestamp,value
    events = pd.read_csv(filepath + 'events.csv',parse_dates=['timestamp'],dtype={'patient_id':int})
    #Columns in mortality_event.csv - patient_id,timestamp,label
    mortality = pd.read_csv(filepath + 'mortality_events.csv',parse_dates=['timestamp'],dtype={'patient_id': int,'label':int})
    #Columns in event_feature_map.csv - idx,event_id
    feature_map = pd.read_csv(filepath + 'event_feature_map.csv',dtype={'idx':int})

    return events, mortality, feature_map


def calculate_index_date(events, mortality, deliverables_path):
    
    '''
    TODO: This function needs to be completed.

    Refer to instructions in Q3 a

    Suggested steps:
    1. Create list of patients alive ( mortality_events.csv only contains information about patients deceased)
    2. Split events into two groups based on whether the patient is alive or deceased
    3. Calculate index date for each patient
    
    IMPORTANT:
    Save indx_date to a csv file in the deliverables folder named as etl_index_dates.csv. 
    Use the global variable deliverables_path while specifying the filepath. 
    Each row is of the form patient_id, indx_date.
    The csv file should have a header 
    For example if you are using Pandas, you could write: 
        indx_date.to_csv(deliverables_path + 'etl_index_dates.csv', columns=['patient_id', 'indx_date'], index=False)

    Return indx_date
    '''
    #Group Alive and Dead Patients
    merged_df = pd.merge(events, mortality, on='patient_id', how='left')
    merged_df['timestamp_x'] = pd.to_datetime(merged_df['timestamp_x'])

    dead_patients = merged_df[merged_df['patient_id'].isin(mortality['patient_id'])] #Grouped dead patients
    alive_patients = merged_df[~merged_df['patient_id'].isin(mortality['patient_id'])] #Grouped alive patients
    #--------------------------------------------------------------------------------------------------------
   #Create Index Date--#CREATE LEAP YEAR CHECK FOR FEBUARY FOR DEAD PATIENTS(NOT DONE)
   #Alive Patients
    alive_index_date = alive_patients.groupby('patient_id')['timestamp_x'].max().reset_index(name='indx_date')
    #print(alive_patients.columns)

   #test
   # desired_patient_id = 198
   # filtered_alive_patients = alive_patients[alive_patients['patient_id'] == desired_patient_id]
   #test 
   #Dead Patients-
    dead_index_date = mortality.copy()
    dead_index_date['timestamp'] = pd.to_datetime(dead_index_date['timestamp'])
    # Subtract one month from the timestamp column or 30 days in problem statement
    dead_index_date['indx_date'] = dead_index_date['timestamp'] - pd.DateOffset(days=30)
   #concatenate to make index_date df for all patients
    indx_date = pd.concat([dead_index_date, alive_index_date], ignore_index=True)
    indx_date = indx_date.loc[:, ['patient_id', 'indx_date']]  # Only keep these 2 columns
   
    indx_date.to_csv(deliverables_path + 'etl_index_dates.csv', columns=['patient_id', 'indx_date'], index=False)
    
    return indx_date


def filter_events(events, indx_date, deliverables_path):
    
    '''
    TODO: This function needs to be completed.

    Refer to instructions in Q3 b

    Suggested steps:
    1. Join indx_date with events on patient_id
    2. Filter events occuring in the observation window(IndexDate-2000 to IndexDate)
    
    
    IMPORTANT:
    Save filtered_events to a csv file in the deliverables folder named as etl_filtered_events.csv. 
    Use the global variable deliverables_path while specifying the filepath. 
    Each row is of the form patient_id, event_id, value.
    The csv file should have a header 
    For example if you are using Pandas, you could write: 
        filtered_events.to_csv(deliverables_path + 'etl_filtered_events.csv', columns=['patient_id', 'event_id', 'value'], index=False)

    Return filtered_events
    '''
    merged_indx = pd.merge(events, indx_date, on='patient_id', how='left') #left join merge 

    def filter_observation_window(patient_data):
        return patient_data[
           (patient_data['timestamp'] >= (patient_data['indx_date'] - pd.DateOffset(days=2000))) &
           (patient_data['timestamp'] <= patient_data['indx_date'])
       ]
    # Apply the filtering function to each patient group
    observation_window = merged_indx.groupby('patient_id').apply(filter_observation_window)
    observation_window = observation_window.reset_index(drop=True)
    #observation_window=observation_window[observation_window['patient_id']==3462]
    #Prediction Window
    prediction_window = merged_indx[
       (merged_indx['timestamp'] >= merged_indx['indx_date']) &
       (merged_indx['timestamp'] <= merged_indx['indx_date'] + pd.DateOffset(days=30))
    ]
    filtered_events = observation_window
    filtered_events.to_csv(deliverables_path + 'etl_filtered_events.csv', columns=['patient_id', 'event_id', 'value'], index=False)
    return filtered_events


def aggregate_events(filtered_events_df, mortality_df,feature_map_df, deliverables_path):
    
    '''
    TODO: This function needs to be completed.

    Refer to instructions in Q3 c

    Suggested steps:
    1. Replace event_id's with index available in event_feature_map.csv
    2. Remove events with n/a values
    3. Aggregate events using sum and count to calculate feature value
    4. Normalize the values obtained above using min-max normalization(the min value will be 0 in all scenarios)
    
    
    IMPORTANT:
    Save aggregated_events to a csv file in the deliverables folder named as etl_aggregated_events.csv. 
    Use the global variable deliverables_path while specifying the filepath. 
    Each row is of the form patient_id, event_id, value.
    The csv file should have a header .
    For example if you are using Pandas, you could write: 
        aggregated_events.to_csv(deliverables_path + 'etl_aggregated_events.csv', columns=['patient_id', 'feature_id', 'feature_value'], index=False)

    Return filtered_events
    '''
    #Check all unique events
    #duplicates = feature_map[feature_map.duplicated(subset='event_id')]
    #Remove all events in observation_window that contain Null values for 'value' feature before aggregating 
    filtered_events_df = filtered_events_df.dropna(subset=['value'])
    # Create a new DataFrame for aggregated events
    agg_events = pd.DataFrame(columns=['patient_id', 'feature_id', 'feature_value'])

    # Iterate through unique patient_ids
    for patient_id in filtered_events_df['patient_id'].unique():
        # Filter data for the current patient_id
        patient_data = filtered_events_df[filtered_events_df['patient_id'] == patient_id]
     
        # For DIAG and DRUG events, sum the corresponding value columns
        diag_drug_data = patient_data[patient_data['event_id'].str.startswith(('DIAG', 'DRUG'))]
        diag_drug_sum = diag_drug_data.groupby('event_id')['value'].sum().reset_index()
        diag_drug_sum['patient_id'] = patient_id  # Add patient_id column
        diag_drug_sum.rename(columns={'event_id': 'feature_id', 'value': 'feature_value'}, inplace=True)

        # For LAB events, count the occurrences of unique event_ids
        lab_data = patient_data[patient_data['event_id'].str.startswith('LAB')]
        lab_count = lab_data['event_id'].value_counts().reset_index()
        lab_count.columns = ['event_id', 'feature_value']  # Rename columns for clarity
        lab_count['patient_id'] = patient_id  # Add patient_id column
        lab_count.rename(columns = {'event_id': 'feature_id'}, inplace=True)
        # Concatenate the results to agg_events
        agg_events = pd.concat([agg_events, diag_drug_sum, lab_count], ignore_index=True)

    #Replace and map each event_id with feature_id in the event_feature_map.csv for observation_window datafram
    merged_agg_events = pd.merge(agg_events, feature_map_df, left_on='feature_id', right_on='event_id', how='left')
    # Drop the redundant 'event_id' column from the merge
    merged_agg_events = merged_agg_events.drop('event_id', axis=1)
    merged_agg_events = merged_agg_events.drop('feature_id', axis=1)
    merged_agg_events = merged_agg_events.rename(columns={'idx': 'feature_id'})
    # Reorder columns
    merged_agg_events = merged_agg_events[['patient_id', 'feature_id', 'feature_value']]
    null_values = merged_agg_events['feature_value'].isnull().sum()


    # Iterate through unique feature_ids
    for feature_id in merged_agg_events['feature_id'].unique():
        # Filter data for the current feature_id
        feature_data = merged_agg_events[merged_agg_events['feature_id'] == feature_id]

        # Calculate the maximum feature_value for the current feature_id
        max_feature_value = feature_data['feature_value'].max()
        
        if max_feature_value != 0:  # avoids division by zero
            # Normalize the feature_value column by dividing each value by the max_feature_value
            merged_agg_events.loc[merged_agg_events['feature_id'] == feature_id, 'feature_value'] = (
                merged_agg_events.loc[merged_agg_events['feature_id'] == feature_id, 'feature_value'] / max_feature_value
            ).round(6)

        if merged_agg_events['feature_value'].isnull().any():
            print(f"NaN values found after normalization for feature_id {feature_id}")
    # Format feature_value to always have 6 decimal places
    merged_agg_events['feature_value'] = merged_agg_events['feature_value'].apply(lambda x: format(x, '.6f'))

# Drop rows with NaN values after normalization
    normalized = merged_agg_events.dropna(subset=['feature_value'])

        
    null_values_normalized = normalized['feature_value'].isnull().sum()
    patient_check = normalized[normalized['patient_id']==19]
    #test
    count_patients_value=normalized['patient_id'].nunique()
    
    
    
    aggregated_events = normalized
    aggregated_events.to_csv(deliverables_path + 'etl_aggregated_events.csv', columns=['patient_id', 'feature_id', 'feature_value'], index=False,header=True)
    
    return aggregated_events

def create_features(events, mortality, feature_map):
    
    deliverables_path = '../deliverables/'

    #Calculate index date
    indx_date = calculate_index_date(events, mortality, deliverables_path)

    #Filter events in the observation window
    filtered_events = filter_events(events, indx_date,  deliverables_path)
    
    #Aggregate the event values for each patient 
    aggregated_events = aggregate_events(filtered_events, mortality, feature_map, deliverables_path)

    '''
    TODO: Complete the code below by creating two dictionaries - 
    1. patient_features :  Key - patient_id and value is array of tuples(feature_id, feature_value)
    2. mortality : Key - patient_id and value is mortality label
    '''
    patient_features = {}

    # Iterate through unique patient_ids
    for patient_id in aggregated_events['patient_id'].unique():
        # Filter data for the current patient_id
        patient_data = aggregated_events[aggregated_events['patient_id'] == patient_id]
        
        # Create a list of tuples (feature_id, feature_value) for the current patient
        features = list(zip(patient_data['feature_id'], patient_data['feature_value']))
        
        # Assign the list of tuples to the patient_id key in the dictionary
        patient_features[patient_id] = features

    dead_patient_id = mortality['patient_id'] #decesed patient_id
    # Create an empty dictionary for mortality
    mortality = {}

    # Iterate through rows in normalized DataFrame
    for index, row in aggregated_events.iterrows():
        patient_id = row['patient_id']

        # Check if patient_id is in mortality_df
        if patient_id in dead_patient_id.values:
            mortality_status = 1
        else:
            mortality_status = 0

        # Update mortality dictionary
        mortality[patient_id] = mortality_status
    #Stock
    #patient_features = {}
    #mortality = {}
    #Stock
    return patient_features, mortality

def save_svmlight(patient_features, mortality, op_file, op_deliverable):
    
    '''
    TODO: This function needs to be completed

    Refer to instructions in Q3 d

    Create two files:
    1. op_file - which saves the features in svmlight format. (See instructions in Q3d for detailed explanation)
    2. op_deliverable - which saves the features in following format:
       patient_id1 label feature_id:feature_value feature_id:feature_value feature_id:feature_value ...
       patient_id2 label feature_id:feature_value feature_id:feature_value feature_id:feature_value ...  
    
    Note: Please make sure the features are ordered in ascending order, and patients are stored in ascending order as well.     
    '''
    with open(op_file, 'wb') as svm_file, open(op_deliverable, 'wb') as deliverable_file:
    # Sort patient_features based on patient_id in ascending order
        sorted_patient_features = sorted(patient_features.items(), key=lambda x: x[0])

        for patient_id, features in sorted_patient_features:
        # Get the mortality label for the current patient
            mortality_label = mortality.get(patient_id, 0)  # Default to 0 if not found

        # Sort features by feature number (assuming features is a list of tuples)
            sorted_features = sorted(features, key=lambda x: x[0])

        # Create the SVM-Light formatted line excluding features with value 0
            svm_line = f"{mortality_label:.0f} {' '.join([f'{feature[0]:.0f}:{feature[1]:.6f}' for feature in sorted_features])} \r\n"
        
        # Write the line to the SVM file
            svm_file.write(bytes(svm_line, 'UTF-8'))

        # Create the line for op_deliverable
            deliverable_line = f"{patient_id:.0f} {mortality_label:.0f} {' '.join([f'{feature[0]:.0f}:{feature[1]:.6f}' for feature in sorted_features])} \r\n"

        # Write the line to the deliverable file
            deliverable_file.write(bytes(deliverable_line, 'UTF-8'))

    
    #stock
    #deliverable1 = open(op_file, 'wb')
    #deliverable2 = open(op_deliverable, 'wb')
    #deliverable1.write(bytes((""),'UTF-8')); #Use 'UTF-8'
    #deliverable2.write(bytes((""),'UTF-8'));
    #stockend





# patient_features = {2293.0: [(2741.0, 1.0), (2751.0, 1.0), (2760.0, 1.0), (2841.0, 1.0), (2880.0, 1.0), (2914.0, 1.0), (2948.0, 1.0), (3008.0, 1.0), (3049.0, 1.0), (1193.0, 1.0), (1340.0, 1.0), (1658.0, 1.0), (1723.0, 1.0), (2341.0, 1.0), (2414.0, 1.0)]}
# mortality = {2293.0: 1}
# etl.save_svmlight(patient_features, mortality, '../deliverables/features_svmlight.train', '../deliverables/features.train')
 

VALIDATION_FEATURES = "../tests/features_svmlight.train"
VALIDATION_DELIVERABLE = "../tests/features.train"
patient_features = {2293.0: [(2741.0, 1.0), (2751.0, 1.0), (2760.0, 1.0), (2841.0, 1.0), (2880.0, 1.0), (2914.0, 1.0), (2948.0, 1.0), (3008.0, 1.0), (3049.0, 1.0), (1193.0, 1.0), (1340.0, 1.0), (1658.0, 1.0), (1723.0, 1.0), (2341.0, 1.0), (2414.0, 1.0)]}
mortality = {2293.0: 1}
save_svmlight(patient_features, mortality, VALIDATION_FEATURES, VALIDATION_DELIVERABLE)
with open("../tests/expected_features.train", 'r', encoding='utf-8') as file:
    content = file.read()
    for char in content:
        if char == '\n':
            print('\\n')
        elif char == '\r':
            print('\\r')
        else:
            print(char, end='')
with open("../tests/features.train", 'r', encoding='utf-8') as file:
    content = file.read()
    for char in content:
        if char == '\n':
            print('\\n')
        elif char == '\r':
            print('\\r')
        else:
            print(char, end='')
            

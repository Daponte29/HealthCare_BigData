import utils
from etl import aggregate_events, create_features
import numpy as np
import pandas as pd
from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import *

'''
Note that for the test data, all events are already filtered such that they fall in the observation window of their respective patients. Thus, if you were to generate features similar to those you constructed in code/etl.py for the test data, all you have to do is aggregate events for each patient.
IMPORTANT: Store your test data features in a file called "test_features.txt" where each line has the
patient_id followed by a space and the corresponding feature in sparse format.
Eg of a line:
60 971:1.000000 988:1.000000 1648:1.000000 1717:1.000000 2798:0.364078 3005:0.367953 3049:0.013514
Here, 60 is the patient id and 971:1.000000 988:1.000000 1648:1.000000 1717:1.000000 2798:0.364078 3005:0.367953 3049:0.013514 is the feature for the patient with id 60.

Save the file as "test_features.txt" and save it inside the folder deliverables

input:
output: X_train,Y_train,X_test
'''
def my_features():
    # TODO: complete this

    # Using same features as in etl.py code. NOT CHANGING FEATURES
    mortality_filepath = '../data/train/mortality_events.csv'

    # Using same features as in etl.py code. NOT CHANGING FEATURES but using data from 'test' folder
    filepath_train ='../data/train'
    filepath_test = '../data/test'
    # Issue with agg_event so make deliverable path so making agg_events_test save to here for now in the same folder/directory as this test code for final my_model HW1
    # Don't really need this file as this was only for etl csv save part only to make the function work for my_model part
    agg_path = '../Predicting Mortality - Classifier Models'
    # End
    # load all raw files to dataframes
    events_train = pd.read_csv(filepath_train + '//events.csv')
    events_train = pd.DataFrame(events_train)
    events_test = pd.read_csv(filepath_test + '//events.csv')
    events_test = pd.DataFrame(events_test)
    mortality = pd.read_csv(mortality_filepath)
    mortality = pd.DataFrame(mortality)
    feature_map_train = pd.read_csv(filepath_train + '//event_feature_map.csv', dtype={'idx': int})
    feature_map_train = pd.DataFrame(feature_map_train)
    feature_map_test = pd.read_csv(filepath_test + '//event_feature_map.csv', dtype={'idx': int})
    feature_map_test = pd.DataFrame(feature_map_test)
    # --------------
    # Create the test_features.txt for X_test using the data/test/events.csv and data/test/event_feature_map.csv

    agg_events_test = aggregate_events(events_test, mortality, feature_map_test, agg_path)
    # got agg_events_test now start making the features needed for the model
    patient_features = {}

    # Iterate through unique patient_ids
    for patient_id in agg_events_test['patient_id'].unique():
        # Filter data for the current patient_id
        patient_data = agg_events_test[agg_events_test['patient_id'] == patient_id]

        # Create a list of tuples (feature_id, feature_value) for the current patient
        features = list(zip(patient_data['feature_id'], [float(value) for value in patient_data['feature_value']]))

        # Assign the list of tuples to the patient_id key in the dictionary
        patient_features[patient_id] = features

    dead_patient_id = mortality['patient_id']  # deceased patient_id
    # Create an empty dictionary for mortality
    mortality = {}

    # Iterate through rows in normalized DataFrame
    for index, row in agg_events_test.iterrows():
        patient_id = row['patient_id']

        # Check if patient_id is in mortality_df
        if patient_id in dead_patient_id.values:
            mortality_status = 1
        else:
            mortality_status = 0

        # Update mortality dictionary
        mortality[patient_id] = mortality_status

    # Created both dictionaries now combine and save them for features in an svm light file to make X_train,Y_train data
    svm_file_path ='../output/test_features.txt'
    # Sort patient_features based on patient_id in ascending order
    sorted_patient_features = sorted(patient_features.items(), key=lambda x: x[0])

    with open(svm_file_path, 'wb') as svm_file:
        for patient_id, features in sorted_patient_features:
            # Get the mortality label for the current patient
            mortality_label = mortality.get(patient_id, 0)  # Default to 0 if not found

            # Sort features by feature number (assuming features is a list of tuples)
            sorted_features = sorted(features, key=lambda x: x[0])

            # Create the SVM-Light formatted line excluding features with value 0
            svm_line = f"{patient_id:.0f} {' '.join([f'{feature[0]:.0f}:{feature[1]:.6f}' for feature in sorted_features])} \r\n"

            # Write the line to the SVM file
            svm_file.write(bytes(svm_line, 'UTF-8'))
 

    X_train, Y_train = utils.get_data_from_svmlight('../output/features_svmlight.train' ) 
    X_test, Y_test = utils.get_data_from_svmlight('../data/features_svmlight.validate')

    return X_train, Y_train, X_test


'''
You can use any model you wish.

input: X_train, Y_train, X_test
output: Y_pred
'''
def my_classifier_predictions(X_train, Y_train, X_test):
    RANDOM_STATE = 545510477  
    X_test, Y_test = utils.get_data_from_svmlight('../data/features_svmlight.validate')
    # Logistic Regression
    param_grid_lr = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga']
    }
    logistic_regression_model = LogisticRegression(random_state=RANDOM_STATE)
    grid_search_lr = GridSearchCV(logistic_regression_model, param_grid_lr, cv=5, scoring='roc_auc', n_jobs=-1)
    grid_search_lr.fit(X_train, Y_train)
    best_lr_model = grid_search_lr.best_estimator_

    # SVM
    param_grid_svm = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto']
    }
    svm_model = SVC(probability=True, random_state=RANDOM_STATE)
    grid_search_svm = GridSearchCV(svm_model, param_grid_svm, cv=5, scoring='roc_auc', n_jobs=-1)
    grid_search_svm.fit(X_train, Y_train)
    best_svm_model = grid_search_svm.best_estimator_

    # Decision Tree
    param_grid_dt = {
        'max_depth': [None, 5, 10, 15, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    decision_tree_model = DecisionTreeClassifier(random_state=RANDOM_STATE)
    grid_search_dt = GridSearchCV(decision_tree_model, param_grid_dt, cv=5, scoring='roc_auc', n_jobs=-1)
    grid_search_dt.fit(X_train, Y_train)
    best_dt_model = grid_search_dt.best_estimator_

    # Gradient Boosting
    param_grid_gb = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 4, 5],
        'subsample': [0.8, 1.0]
    }
    gradient_boosting_model = GradientBoostingClassifier(random_state=RANDOM_STATE)
    grid_search_gb = GridSearchCV(gradient_boosting_model, param_grid_gb, cv=5, scoring='roc_auc', n_jobs=-1)
    grid_search_gb.fit(X_train, Y_train)
    best_gb_model = grid_search_gb.best_estimator_

    # Compare models
    models = {
        'Logistic Regression': best_lr_model,
        'SVM': best_svm_model,
        'Decision Tree': best_dt_model,
        'Gradient Boosting': best_gb_model
    }

    for name, model in models.items():
        # Cross-validation score
        cv_score = cross_val_score(model, X_train, Y_train, cv=5, scoring='roc_auc')
        print(f'{name} - Cross Validation Score: {np.mean(cv_score)}')

        # AUC on test set
        Y_pred_proba = model.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(Y_test, Y_pred_proba)
        print(f'{name} - AUC on Test Set: {auc_score}')

    # Return the predictions of the best model (you can choose the best based on cross-validation scores)
    best_model = models[max(models, key=lambda k: np.mean(cross_val_score(models[k], X_train, Y_train, cv=5, scoring='roc_auc')))]
    Y_pred = best_model.predict(X_test)

    return Y_pred


def main():
    X_train, Y_train, X_test = my_features()
    Y_pred = my_classifier_predictions(X_train, Y_train, X_test)

    # Read the patient IDs from the test features file
    test_patient_ids = []
    with open("../output/test_features.txt", "r") as f:
        for line in f:
            # Extract the patient ID from each line
            patient_id = int(line.split()[0])
            test_patient_ids.append(patient_id)

    # Create a DataFrame with patient IDs and predicted labels
    predictions_df = pd.DataFrame({'Patient_ID': test_patient_ids, 'Predicted_Label': Y_pred})

    # Print the first few rows of the DataFrame to verify its contents
    print(predictions_df.head())

    # Save the DataFrame to a CSV file
    predictions_df.to_csv("../output/my_predictions.csv", index=False)

if __name__ == "__main__":
    main()

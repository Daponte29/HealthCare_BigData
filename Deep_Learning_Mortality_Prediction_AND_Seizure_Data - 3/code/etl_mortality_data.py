import os
import pickle
import pandas as pd
import numpy as np
##### DO NOT MODIFY OR REMOVE THIS VALUE #####
checksum = '169a9820bbc999009327026c9d76bcf1'
##### DO NOT MODIFY OR REMOVE THIS VALUE #####
#FINAL SUBB
PATH_TRAIN = "../data/mortality/train/"
PATH_VALIDATION = "../data/mortality/validation/"
PATH_TEST = "../data/mortality/test/"
PATH_OUTPUT = "../data/mortality/processed/"


def convert_icd9(icd9_object): #extract each string in ICD9 and use in createdata set func.
    """
    :param icd9_object: ICD-9 code (Pandas/Numpy object).
    :return: extracted main digits of ICD-9 code
    """
    icd9_str = str(icd9_object)
    
    if icd9_str.startswith('E'):
        converted = icd9_str[:4]    
    else:
        converted = icd9_str[:3]
    
    return converted


def build_codemap(df_icd9, transform):
    """
    :return: Dict of code map {main-digits of ICD9: unique feature ID}
    """
    df_icd9 = df_icd9.dropna(subset=['ICD9_CODE'])
                             
    
    df_digits = df_icd9['ICD9_CODE'].apply(transform)
    codemap = {code: idx for idx, code in enumerate(df_digits.unique())}
    return codemap
                        



def create_dataset(path, codemap, transform):
    """
    :param path: path to the directory containing raw files.
    :param codemap: 3-digit ICD-9 code feature map
    :param transform: e.g., convert_icd9
    :return: List(patient IDs), List(labels), Visit sequence data as a List of List of List.
    """
    # TODO: 1. Load data from the three CSV files
    # TODO: Loading the mortality file is shown as an example below. Load two other files also.
    df_mortality = pd.read_csv(os.path.join(path,"MORTALITY.csv"))
    df_diagnosis = pd.read_csv(os.path.join(path,"DIAGNOSES_ICD.csv"))
    df_admissions = pd.read_csv(os.path.join(path,"ADMISSIONS.csv"))
    
    df_diagnosis = df_diagnosis.dropna(subset=['ICD9_CODE'])
    
    
    df_diagnosis['FEATURE_ID'] = df_diagnosis['ICD9_CODE'].apply(transform).map(codemap)
    df_diagnosis.drop('ICD9_CODE', axis = 1, inplace = True)
    
    df_groupe_diagnosis = df_diagnosis.groupby(['HADM_ID'])['FEATURE_ID'].apply(list).reset_index()
    
    df_visits = pd.merge(df_admissions[['HADM_ID', 'SUBJECT_ID', 'ADMITTIME']], df_groupe_diagnosis, on = 'HADM_ID')
    df_visits['ADMITTIME'] = pd.to_datetime(df_visits['ADMITTIME'])
    df_visits = df_visits.sort_values(by=['SUBJECT_ID', 'ADMITTIME'])
    grouped_visits = df_visits.groupby('SUBJECT_ID')['FEATURE_ID'].apply(list)
    
    
    seq_data = grouped_visits.tolist()
    patient_ids = grouped_visits.index.tolist()
    
    df_labels = pd.merge(df_mortality[['SUBJECT_ID', 'MORTALITY']], df_visits[['SUBJECT_ID']].drop_duplicates(), on = 'SUBJECT_ID')
    labels = df_labels.sort_values(by ='SUBJECT_ID')['MORTALITY'].tolist()
       
    return patient_ids, labels, seq_data



def main():
    # Build a code map from the train set
    print("Build feature id map")
    df_icd9 = pd.read_csv(os.path.join(PATH_TRAIN, "DIAGNOSES_ICD.csv"), usecols=["ICD9_CODE"])
    codemap = build_codemap(df_icd9, convert_icd9)
    os.makedirs(PATH_OUTPUT, exist_ok=True)
    pickle.dump(codemap, open(os.path.join(PATH_OUTPUT, "mortality.codemap.train"), 'wb'), pickle.HIGHEST_PROTOCOL)

    # Train set
    print("Construct train set")
    train_ids, train_labels, train_seqs = create_dataset(PATH_TRAIN, codemap, convert_icd9)

    pickle.dump(train_ids, open(os.path.join(PATH_OUTPUT, "mortality.ids.train"), 'wb'), pickle.HIGHEST_PROTOCOL)
    pickle.dump(train_labels, open(os.path.join(PATH_OUTPUT, "mortality.labels.train"), 'wb'), pickle.HIGHEST_PROTOCOL)
    pickle.dump(train_seqs, open(os.path.join(PATH_OUTPUT, "mortality.seqs.train"), 'wb'), pickle.HIGHEST_PROTOCOL)
    #TEST--------
    print(codemap)
    inverse_dict = {v: k for k, v in codemap.items()}
    print(inverse_dict)
    print(train_ids[5])
    print("-------------")
    print(train_labels[5])
    print("------------")
    print(train_seqs[5])
    print("------------")
    code_keys = [inverse_dict.get(ids) for visits in train_seqs[5] for ids in visits]
    print(code_keys)
    # Get the corresponding values from the codemap based on the keys
    feature_values = [codemap[key] for key in code_keys]

    print(feature_values)
    #END TEST-----------
    # Validation set
    print("Construct validation set")
    validation_ids, validation_labels, validation_seqs = create_dataset(PATH_VALIDATION, codemap, convert_icd9)

    pickle.dump(validation_ids, open(os.path.join(PATH_OUTPUT, "mortality.ids.validation"), 'wb'), pickle.HIGHEST_PROTOCOL)
    pickle.dump(validation_labels, open(os.path.join(PATH_OUTPUT, "mortality.labels.validation"), 'wb'), pickle.HIGHEST_PROTOCOL)
    pickle.dump(validation_seqs, open(os.path.join(PATH_OUTPUT, "mortality.seqs.validation"), 'wb'), pickle.HIGHEST_PROTOCOL)

    # Test set
    print("Construct test set")
    test_ids, test_labels, test_seqs = create_dataset(PATH_TEST, codemap, convert_icd9)

    pickle.dump(test_ids, open(os.path.join(PATH_OUTPUT, "mortality.ids.test"), 'wb'), pickle.HIGHEST_PROTOCOL)
    pickle.dump(test_labels, open(os.path.join(PATH_OUTPUT, "mortality.labels.test"), 'wb'), pickle.HIGHEST_PROTOCOL)
    pickle.dump(test_seqs, open(os.path.join(PATH_OUTPUT, "mortality.seqs.test"), 'wb'), pickle.HIGHEST_PROTOCOL)

    print("Complete!")


if __name__ == '__main__':
    main()


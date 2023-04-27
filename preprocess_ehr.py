
import pickle
import pandas as pd
import numpy as np
import datetime

pd.set_option('display.width', 2000)

def preprocess_ehr(file):
    pd.set_option('display.width',2000)

    ehr = pd.read_csv(file, names=['ID','Diagnosis','ICD9','DiagnosisDate','AdmissionID'])
    ehr = ehr.dropna()

    # Calculate IDC9 code frequencies
    diagnosis_counts = ehr['Diagnosis'].value_counts()
    print(len(diagnosis_counts))
    # idc9_counts_percentiles = idc9_counts.quantile([0.1, 1])

    # Filter out IDC9 codes below the 10th percentile
    diagnosis_above_percentile = diagnosis_counts[diagnosis_counts > 100].index
    print(len(diagnosis_above_percentile))
    ehr = ehr[ehr['Diagnosis'].isin(diagnosis_above_percentile)]
    vocab_df = ehr['Diagnosis'].drop_duplicates().reset_index().drop('index', axis=1).reset_index().set_index('Diagnosis')
    ehr['encoded_diagnosis'] = ehr['Diagnosis'].apply(lambda d: vocab_df.loc[d])
    print(ehr['encoded_diagnosis'][:10],vocab_df)


    unique_ids = ehr['ID'].unique()
    df_unique = pd.DataFrame({'ID':unique_ids})
    df_unique['DiagnosisDate'] = ehr['DiagnosisDate'].max()
    df_unique['encoded_diagnosis'] = -1
    df_unique['AdmissionID'] = df_unique['ID']

    # IDC9_for_dict = list(ehr['IDC9'].unique())
    # IDC9_dict = {index: code for code, index in enumerate(IDC9_for_dict)}
    # ehr['IDC9'] = ehr['IDC9'].map(IDC9_dict)

    ehr = pd.concat([ehr,df_unique],ignore_index = True)
    ehr_grouped = ehr.groupby(['ID','AdmissionID']).\
         agg({'encoded_diagnosis': 'unique', 'ICD9': 'unique','DiagnosisDate':'max'}).reset_index()
    ehr_grouped_sorted = ehr_grouped.sort_values(by = 'DiagnosisDate',ascending = True)
    ehr_grouped_sorted = ehr_grouped_sorted.groupby('ID').agg({'encoded_diagnosis':lambda x:x.tolist()}).reset_index()
    list_visits_by_person = ehr_grouped_sorted['encoded_diagnosis'].values.tolist()

    list_visits = [visits for person in list_visits_by_person for visits in person]

    for i in range(len(list_visits)):
        list_visits[i] = list(list_visits[i])
    print(len(vocab_df))
    print(vocab_df[:10])
    print(list_visits[:20])
    return list_visits,vocab_df



def save_data(filename,file):
    data,vocab = preprocess_ehr(file)
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    vocab.to_csv('./data/vocab_med2vec_new_large_data_no_rare_diagnosis')

if __name__ == '__main__':
    file = 'data_real_not_to_export.csv'
    file_name_for_model = 'data/data_large/med2vec.seqs'
    save_data(file_name_for_model,file)
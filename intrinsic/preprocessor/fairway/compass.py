import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
import math
import numpy as np

## Load dataset
name = 'compas-scores-two-years.csv'

path = 'data/fairway/' + name
missing_values = ["?"]
dataset_orig = pd.read_csv(path, na_values=missing_values)



## Drop categorical features
## Removed two duplicate coumns - 'decile_score','priors_count'
dataset_orig = dataset_orig.drop(['id','name','first','last','compas_screening_date','dob','age','juv_fel_count','decile_score','juv_misd_count','juv_other_count','days_b_screening_arrest','c_jail_in','c_jail_out','c_case_number','c_offense_date','c_arrest_date','c_days_from_compas','c_charge_desc','is_recid','r_case_number','r_charge_degree','r_days_from_arrest','r_offense_date','r_charge_desc','r_jail_in','r_jail_out','violent_recid','is_violent_recid','vr_case_number','vr_charge_degree','vr_offense_date','vr_charge_desc','type_of_assessment','decile_score','score_text','screening_date','v_type_of_assessment','v_decile_score','v_score_text','v_screening_date','in_custody','out_custody','start','end','event'],axis=1)

## Drop NULL values
dataset_orig = dataset_orig.dropna()


## Change symbolics to numerics
dataset_orig['sex'] = np.where(dataset_orig['sex'] == 'Female', 1, 0)
dataset_orig['race'] = np.where(dataset_orig['race'] != 'Caucasian', 0, 1)
dataset_orig['priors_count'] = np.where((dataset_orig['priors_count'] >= 1 ) & (dataset_orig['priors_count'] <= 3), 3, dataset_orig['priors_count'])
dataset_orig['priors_count'] = np.where(dataset_orig['priors_count'] > 3, 4, dataset_orig['priors_count'])
dataset_orig['age_cat'] = np.where(dataset_orig['age_cat'] == 'Greater than 45',45,dataset_orig['age_cat'])
dataset_orig['age_cat'] = np.where(dataset_orig['age_cat'] == '25 - 45', 25, dataset_orig['age_cat'])
dataset_orig['age_cat'] = np.where(dataset_orig['age_cat'] == 'Less than 25', 0, dataset_orig['age_cat'])
dataset_orig['c_charge_degree'] = np.where(dataset_orig['c_charge_degree'] == 'F', 1, 0)


## Rename class column
dataset_orig.rename(index=str, columns={"two_year_recid": "Probability"}, inplace=True)

## Here did not rec means 0 is the favorable lable
# dataset_orig['Probability'] = np.where(dataset_orig['Probability'] == 0, 1, 0)

scaler = MinMaxScaler()
dataset_orig = pd.DataFrame(scaler.fit_transform(dataset_orig),columns = dataset_orig.columns)

dataset_orig_x = dataset_orig.loc[:, dataset_orig.columns != 'Probability']
print(dataset_orig_x)
dataset_orig_x.to_csv('processed_data/fairway/processed_' + name, index=False)

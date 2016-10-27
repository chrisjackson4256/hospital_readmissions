from __future__ import division
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

path_to_data = "/Users/chris.jackson/models/acrr/data/"

# import hospital/patient data
hsp = pd.read_csv(path_to_data + "hsp_readmissions.csv")

# read in vitals
vitals_min_max = pd.read_csv(path_to_data + "vitals_min_max_adm_24hr.csv")
vitals_min_max.drop(['Unnamed: 0'], axis=1, inplace=True)

# read in labs
labs_min_max = pd.read_csv(path_to_data + "labs_min_max_adm_24hr.csv")

# combine vitals and labs
vitals_labs = pd.merge(left=vitals_min_max, right=labs_min_max,
                       how='left', on=['visit_id'])

# combine admin and vitals/labs
vitals_labs.drop_duplicates(subset=['visit_id'], inplace=True)
full_df = pd.merge(left=hsp, right=vitals_labs, how='left', on=['visit_id'])
full_df.drop_duplicates(subset=['visit_id'], inplace=True)

continuous_list = ['so_age', 'hi_edvis_sum_365d', 'hi_hspadm_sum_365d',
                   'vi_sysbp_max_adm_24', 'vi_sysbp_min_adm_24',
                   'vi_diasbp_max_adm_24','vi_diasbp_min_adm_24',
                   'vi_pulse_max_adm_24', 'vi_pulse_min_adm_24',
                   'vi_resp_max_adm_24', 'vi_resp_min_adm_24',
                   'vi_spo2_max_adm_24', 'vi_spo2_min_adm_24',
                   'vi_temp_max_adm_24', 'vi_temp_min_adm_24',
                   'la_hct_max_adm_24', 'la_hct_min_adm_24',
                   'la_hemo_max_adm_24','la_hemo_min_adm_24',
                   'la_sod_max_adm_24', 'la_sod_min_adm_24',
                   'la_trop_max_adm_24', 'la_trop_min_adm_24',
                   'la_wbc_max_adm_24', 'la_wbc_min_adm_24',
                   'la_alb_max_adm_24', 'la_alb_min_adm_24',
                   'la_lact_max_adm_24', 'la_lact_min_adm_24',
                   'la_gluc_max_adm_24', 'la_gluc_min_adm_24',
                   'la_pot_max_adm_24', 'la_pot_min_adm_24',
                   'la_bun_max_adm_24', 'la_bun_min_adm_24',
                   'la_bnp_max_adm_24', 'la_bnp_min_adm_24',
                   'la_inr_max_adm_24', 'la_inr_min_adm_24',
                   'la_phart_max_adm_24', 'la_phart_min_adm_24',
                   'la_phven_max_adm_24', 'la_phven_min_adm_24',
                   'la_creat_max_adm_24', 'la_creat_min_adm_24',
                   'la_bcr_max_adm_24', 'la_bcr_min_adm_24',
                   'la_plats_max_adm_24', 'la_plats_min_adm_24',
                   'la_po2art_max_adm_24', 'la_po2art_min_adm_24',
                   'la_lymphabs_max_adm_24', 'la_lymphabs_min_adm_24',
                   'la_calc_max_adm_24','la_calc_min_adm_24',
                   'la_aniongap_max_adm_24', 'la_aniongap_min_adm_24',
                   'la_mcv_max_adm_24', 'la_mcv_min_adm_24',
                   'la_ast_max_adm_24','la_ast_min_adm_24',
                   'la_tbilib_max_adm_24', 'la_tbilib_min_adm_24',
                   'la_proth_max_adm_24', 'la_proth_min_adm_24',
                   'la_alt_max_adm_24','la_alt_min_adm_24']

# impute missing values and normalize data
min_max_scaler = MinMaxScaler()
for feat in continuous_list:
    full_df[feat] = pd.to_numeric(full_df[feat], errors='coerce')

    # impute with median values
    full_df[feat].fillna(full_df[feat].median(), inplace=True)

    # scale s.t. feat ranges from 0 to 1
    full_df[feat] = min_max_scaler.fit_transform(full_df[feat])


# convert boolean columns to numeric (for model-building reasons)
bool_columns = ['re30_0hr', 'icd9or10_ami_365d_inp', 'icd9or10_chf_365d_inp',
                'icd9or10_pvd_365d_inp', 'icd9or10_cvd_365d_inp',
                'icd9or10_dementia_365d_inp', 'icd9or10_copd_365d_inp',
                'icd9or10_ctd_365d_inp', 'icd9or10_peptulcer_365d_inp',
                'icd9or10_mildliver_365d_inp', 'icd9or10_diabwocc_365d_inp',
                'icd9or10_hemi_365d_inp', 'icd9or10_renal_365d_inp',
                'icd9or10_diabwcc_365d_inp', 'icd9or10_tumor_wo_meta_365d_inp',
                'icd9or10_leukemia_365d_inp', 'icd9or10_lymphoma_365d_inp',
                'icd9or10_sevmodliver_365d_inp',
                'icd9or10_metasolidtumor_365d_inp','icd9or10_hivaids_365d_inp']

for col in bool_columns:
    full_df[col] = full_df[col].astype(int)

# reorder columns
full_df = full_df[bool_columns + ['admdt', 'pt_id'] + continuous_list]

# keep only a handful of features (for the DNN model)
full_df = full_df[['re30_0hr', 'so_age', 'hi_edvis_sum_365d',
                   'hi_hspadm_sum_365d', 'vi_sysbp_max_adm_24',
                   'vi_diasbp_min_adm_24', 'vi_pulse_min_adm_24',
                   'admdt', 'pt_id']]

# split data into training/validation/testing sets
# define time periods
train_start = '2015-01-01 00:00:00'
train_stop = '2015-08-31 23:59:59'

test_start = '2015-09-01 00:00:00'
test_stop = '2016-01-06 23:59:59'

# save a holdout set
holdout_start = '2014-01-01 00:00:00'
holdout_stop = '2014-12-31 23:59:59'
holdout = full_df[(full_df['admdt'] >= holdout_start) & \
                  (full_df['admdt'] <= holdout_stop)]
holdout.drop('pt_id', axis=1, inplace=True)
holdout.to_csv(path_to_data + "holdout_readmissions.csv", index=False)

# define train/test sets
train = full_df[(full_df['admdt'] >= train_start) & \
                (full_df['admdt'] <= train_stop)]
test = full_df[(full_df['admdt'] >= test_start) & \
               (full_df['admdt'] <= test_stop)]

# make sure patients in training set don't show up in test set
train_pt_list = train['pt_id'].tolist()
test_pt_list = test['pt_id'].tolist()
pt_train_test_overlap_list = list(set(train_pt_list) & set(test_pt_list))
test = test[~test['pt_id'].isin(pt_train_test_overlap_list)]

# we don't need pt_id anymore
train.drop(["pt_id", "admdt"], axis=1, inplace=True)
test.drop(["pt_id", "admdt"], axis=1, inplace=True)

train.to_csv(path_to_data + "train_readmissions.csv", index=False)
test.to_csv(path_to_data + "test_readmissions.csv", index=False)

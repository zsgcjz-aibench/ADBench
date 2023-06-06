import pandas as pd

ac_train = pd.read_csv('origin_data/tfrecord_data/ac_train.csv')
ac_valid = pd.read_csv('origin_data/tfrecord_data/ac_valid.csv')
ac_test = pd.read_csv('origin_data/tfrecord_data/ac_test.csv')
mci_test = pd.read_csv('origin_data/tfrecord_data/mci_test.csv')
smc_test = pd.read_csv('origin_data/tfrecord_data/smc_test.csv')

Merge_info = pd.read_csv('origin_data/ADNIMERGE_without_bad_value.csv')

open_test = pd.concat([ac_test, mci_test, smc_test], axis=0)

final_image_need_process = pd.read_csv('final_image_need_process.csv')
final_image_need_process.rename(columns={'VISCODE2':'VISCODE'}, inplace=True)

ac_train = pd.merge(ac_train, final_image_need_process, on=['RID', 'VISCODE'])
ac_train = pd.merge(ac_train, Merge_info, on=['RID', 'VISCODE'])
ac_train.to_csv('./Open_clinical_setting/train.csv', index=0)

ac_valid = pd.merge(ac_valid, final_image_need_process, on=['RID', 'VISCODE'])
ac_valid = pd.merge(ac_valid, Merge_info, on=['RID', 'VISCODE'])
ac_valid.to_csv('./Open_clinical_setting/val.csv', index=0)

open_test = pd.merge(open_test, final_image_need_process, on=['RID', 'VISCODE'])
open_test = pd.merge(open_test, Merge_info, on=['RID', 'VISCODE'])
open_test.to_csv('./Open_clinical_setting/test.csv', index=0)

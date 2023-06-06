import pandas as pd

ac_train = pd.read_csv('origin_data/tfrecord_data/ac_train.csv')
ac_valid = pd.read_csv('origin_data/tfrecord_data/ac_valid.csv')
ac_test = pd.read_csv('origin_data/tfrecord_data/ac_test.csv')

Merge_info = pd.read_csv('origin_data/ADNIMERGE_without_bad_value.csv')

final_image_need_process = pd.read_csv('final_image_need_process.csv')
final_image_need_process.rename(columns={'VISCODE2':'VISCODE'}, inplace=True)

ac_train = pd.merge(ac_train, final_image_need_process, on=['RID', 'VISCODE'])
ac_train = pd.merge(ac_train, Merge_info, on=['RID', 'VISCODE'])
ac_train.to_csv('./Closed_clinical_setting/train.csv', index=0)

ac_valid = pd.merge(ac_valid, final_image_need_process, on=['RID', 'VISCODE'])
ac_valid = pd.merge(ac_valid, Merge_info, on=['RID', 'VISCODE'])
ac_valid.to_csv('./Closed_clinical_setting/val.csv', index=0)

ac_test = pd.merge(ac_test, final_image_need_process, on=['RID', 'VISCODE'])
ac_test = pd.merge(ac_test, Merge_info, on=['RID', 'VISCODE'])
ac_test.to_csv('./Closed_clinical_setting/test.csv', index=0)

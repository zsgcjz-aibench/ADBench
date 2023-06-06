import pandas as pd
import copy


image_info_path = 'origin_data/image_information.csv'

image_info = pd.read_csv(image_info_path)


image_info.rename(columns={'Visit': 'VISCODE2'}, inplace=True)

#Mode =MRI, sort according to 'RID', 'VISCODE2','Description'
image_info_MRI = image_info.loc[image_info['Modality'] == 'MRI']
image_info_MRI = image_info_MRI.sort_values(by=['RID', 'VISCODE2','Description'])

print("Mode =MRI, len(image_information):",image_info_MRI.shape[0])

#Check for bl blank data in VISCODE column in all visit records of RID. If the RID contains sc or scmri or m03 non-blank data,
# one of sc,scmri or m03 data will be used to complete bl visit data.
def combine_m03_and_sc_into_bl(df, isImage=False, Modality='MRI'):
    """
    :param rID_set: RID of all subjects
    :param sub_df: All MRI data from current RID subjects at m03, bl, sc, and scmri visit records
    :param bl_flag: Whether the current RID subjects has bl data, flags
    :param sc_flag: Whether the current RID subjects has sc, scmri data, markers
    :param m03_flage: Whether the current RID subjects has m03 data, marked
    :return:
    """
    if ('VISCODE2' not in df.columns) and ('VISCODE' in df.columns):
        df.rename(columns={'VISCODE': 'VISCODE2'}, inplace=True)  # Change VISCODE to VISCODE2 in order to uniformly express viscode

    rID_set = df['RID'].drop_duplicates()  #All subjects RID

    for index, row in rID_set.items():
        rRID = row
        if isImage:
            sub_df = df.loc[(df['RID'] == rRID) & (df['Modality'] == Modality) & (
                    (df['VISCODE2'] == 'm03') | (df['VISCODE2'] == 'bl') | (df['VISCODE2'] == 'sc') | (
                    df['VISCODE2'] == 'scmri'))]  #All MRI data of current subjects in m03, bl, sc and scmri visit records were screened

        if sub_df.shape[0] > 0:
            bl_flag = sub_df['VISCODE2'].isin(['bl']).any()  #Determine whether the current subjects has data for bl visit records
            sc_flag = sub_df['VISCODE2'].isin(['sc', 'scmri']).any()
            m03_flage = sub_df['VISCODE2'].isin(['m03']).any()

            if bl_flag and (sc_flag or m03_flage):  # The bl co-exists with one or both of the other two phases, and the data in each column of the bl is recorded as empty
                columns = sub_df.columns.values.tolist()  #Gets the value of the column label for sub_df
                mark = []  #Used to store bl records with empty columns, (index, column name)
                for index_order, row in sub_df.iterrows():
                    visit = row['VISCODE2']
                    if visit == 'bl':   #The patient was recorded to have an empty column during bl follow-up
                        for i in range(0, len(columns)):
                            tmp_value = row[columns[i]]
                            if pd.isnull(tmp_value) or tmp_value == '-4' or tmp_value == '\"\"' or tmp_value == '':
                                mark.append([index_order, columns[i]])

            elif (not bl_flag) and (sc_flag and m03_flage):  # bl does not exist, but sc and m03 exist
                columns = sub_df.columns.values.tolist()
                mark = []
                change_visitcoede = [] #Follow-up records of sc and scmri stage were recorded, and the follow-up viscode of this array was changed to bl later
                for index_order, row in sub_df.iterrows():
                    visit = row['VISCODE2']
                    if visit == 'sc' or visit == 'scmri':
                        change_visitcoede.append(index_order)
                        for i in range(0, len(columns)):
                            tmp_value = row[columns[i]]
                            if pd.isnull(tmp_value) or tmp_value == '-4' or tmp_value == '\"\"' or tmp_value == '':
                                mark.append([index_order, columns[i]])

                for index in change_visitcoede:
                    # print('Before modification：',df.at[index_order, 'VISCODE2'])
                    df.at[index, 'VISCODE2'] = 'bl'
                    # print('After modification：',df.at[index_order, 'VISCODE2'])
            elif (not (bl_flag or m03_flage)) and sc_flag:  # Only the sc phase exists
                for index_order, row in sub_df.iterrows():
                    # print('Before modification：',df.at[index_order, 'VISCODE2'])
                    df.at[index_order, 'VISCODE2'] = 'bl'
                    # print('After modification：',df.at[index_order, 'VISCODE2'])
            elif (not (bl_flag or sc_flag)) and m03_flage:  # Only phase m03 exists
                for index_order, row in sub_df.iterrows():
                    # print('Before modification：',df.at[index_order, 'VISCODE2'])
                    df.at[index_order, 'VISCODE2'] = 'bl'
                    # print('After modification：',df.at[index_order, 'VISCODE2'])

    return df

image_info_MRI = combine_m03_and_sc_into_bl(image_info_MRI,isImage=True,Modality='MRI')
print("When the mode is MRI, the image_information length of the data changed to bl is supplemented by sc,scmri and m03:",image_info_MRI.shape[0])

#df 缺失值填充
def set_missing_value(df):
    #df.where(df.notnull(),'-4')
    df =df.fillna('-4')
    df =df.where(df !='', '-4')
    df =df.where(df != '\"\"', '-4')
    return df

image_info_MRI=set_missing_value(image_info_MRI)

#Example Delete data with Sequence = -4
image_info_MRI = image_info_MRI.loc[image_info_MRI['Sequence'] == 1]
print("The image_information length after deleting data with Sequence = -4:",image_info_MRI.shape[0])

#按'RID', 'VISCODE2','Sequence'去重
image_info_MRI = image_info_MRI.drop_duplicates(subset=['RID', 'VISCODE2','Sequence'])
print("Follow the image_information length after ['RID', 'VISCODE2','Sequence'] viscodeweight:",image_info_MRI.shape[0])
# image_info_MRI.to_csv('data/complete_imfo250.csv', index=False)



#The processed complete_imfo is matched with the model_data table
model_data_path = 'origin_data/OpenClinicalAI_data.csv'
model_data = pd.read_csv(model_data_path)

df3 = pd.merge(model_data, image_info_MRI, on=['RID', 'VISCODE2'])
df3 = df3.drop(df3[df3.SavePath == '-4'].index)
print("The data in the tfrecord is contained in the image_merge_information length:",df3.shape[0])
# df3.to_csv('./df3.csv', index=0)

#Retrieve the file names of all images in the data used for training
filename = df3.loc[:, ['RID', 'VISCODE2', 'SavePath']]
df8 = copy.deepcopy(filename)
for index, row in filename.iterrows():
    # print('row= ', row)
    df8.loc[index, 'SavePath'] = (str(row['SavePath']).split('/')[-1])[:-3] + "npy"

no_process_image = pd.read_csv('origin_data/no_processed.csv')
no_processed_images = no_process_image['SavePath'].values.tolist()
for i in range(len(no_processed_images)):
    df8 = df8.drop(df8[df8['SavePath'] == no_processed_images[i]].index)
df8.to_csv('./final_image_need_process.csv', index=0)

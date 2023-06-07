#-*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import ants
import os
import nibabel as nib  # nii格式一般都会用到这个包
import imageio  # 转换成图像
from PIL import Image
import math
import tensorflow as tf
import copy
import random

random.seed(1)

data_set_path='/data/huangyunyou/ADNI/'

merger_path='/data/huangyunyou/ADNI/ADNIMERGE_without_bad_value.csv'

scale_path='/data/huangyunyou/ADNI/SCALE/'
scale_base=scale_path+'base_scale.csv'
scale_cog=scale_path+'cog_scale.csv'
scale_neuro=scale_path+'neuro_scale.csv'
scale_cog_exame=scale_path+'tmpN/clear_NEUROBAT.csv'
scale_fb=scale_path+'fb_scale.csv'
scale_exame=scale_path+'exam_scale.csv'

bio_path='/data/huangyunyou/ADNI/BIO/'

bio_plasma=bio_path+'plasma_without_bad_value.csv'
bio_serum=bio_path+'serum.csv'
bio_urine=bio_path+'urine_without_bad_value.csv'
bio_csf=bio_path+'csf_without_bad_value.csv'


gen_path='/data/huangyunyou/ADNI/GEN/'
gen_ANDI1GO2=gen_path+'ADNI1GO2_20201020.txt'
gen_ADNI1=gen_path+'ADNI1_20201020.txt'
gen_ADNIGO2=gen_path+'ADNIGO2_20201020.txt'
gen_ADNI3=gen_path+'ADNI3_20201020.txt'


image_path='/data/huangyunyou/ADNI/IMAGE/image_information.csv'



# 图像配准
image_fix_path = '/data/huangyunyou/ADNI/IMAGE/template/'
fixed_T1 = ants.image_read(image_fix_path + 'mni_icbm152_lin_nifti/icbm_avg_152_t1_tal_lin.nii')
fixed_T2 = ants.image_read(image_fix_path + 'mni_icbm152_lin_nifti/icbm_avg_152_t2_tal_lin.nii')
fixed_PD = ants.image_read(image_fix_path + 'mni_icbm152_lin_nifti/icbm_avg_152_pd_tal_lin.nii')

fixed_FDG_PET = ants.image_read(image_fix_path + 'MNI152_PET_1mm.nii')

#由于AV45的模版有问题，利用0填充缺失值
def replaceNaNWithZero(datMat):
    datMat = datMat[:, :, 8:82]
    # 遍历数据集每一个维度
    for i in range(91):
        for j in range(109):
            for l in range(74):
                if np.isnan(datMat[i, j, l]):
                    datMat[i, j, l] = 0
    return datMat  # 最后返回用平均值填充空缺值后的数组

fixed_AV45 = ants.image_read(image_fix_path + '/AV45_Niftii_Templates/CN_AV45_WithSkull.nii')
ftmp = fixed_AV45.numpy()
ftmp = replaceNaNWithZero(ftmp)#去掉包含空值过多的图片，以及给有少量空值的图片补上黑色
fixed_AV45 = ants.from_numpy(ftmp)

miss_imageID_set=set()
miss_imageID_save_path='/data/huangyunyou/ADNI/IMAGE/miss_image_smc.txt'
with open(miss_imageID_save_path, 'a') as f:
    f.write('RID,Viscode,Study Date,Modality,ImageID,Radiopharmaceutical\n')
sample_statistics_information_save_path='/data/huangyunyou/ADNI/sample_statistics.txt'
miss_image_dict={}
registration_save='/data/huangyunyou/ADNI/registration/'
image_array_path='/home/huangyunyou/ADNI/IMAGE/array/'
#DenseNet_save='/home/huangyunyou/ADNI/IMAGE/densnet/'
abnormal_number_viscode_save_path='/data/huangyunyou/ADNI/abnormal_viscode.txt'

data_len_save_path='/data/huangyunyou/ADNI/data_len.txt'
data_len_save_path_2v='/data/huangyunyou/ADNI/data_len_2v.txt'
# 利用 sc 和 m03 的数据补全 bl 阶段的数据
def combine_m03_and_sc_into_bl(df, isImage=False, Modality='MRI'):
    if ('VISCODE2' not in df.columns) and ('VISCODE' in df.columns):
        df.rename(columns={'VISCODE': 'VISCODE2'}, inplace=True)  # 为了统一表达 VISCODE 修改为VISCODE2

    rID_set = df['RID'].drop_duplicates()
    # df = df.set_index(['RID', 'VISCODE2'], drop=False)
    for index, row in rID_set.items():
        rRID = row
        if isImage:
            sub_df = df.loc[(df['RID'] == rRID) & (df['Modality'] == Modality) & (
                    (df['VISCODE2'] == 'm03') | (df['VISCODE2'] == 'bl') | (df['VISCODE2'] == 'sc') | (
                    df['VISCODE2'] == 'scmri'))]
        else:
            sub_df = df.loc[(df['RID'] == rRID) & (
                    (df['VISCODE2'] == 'm03') | (df['VISCODE2'] == 'bl') | (df['VISCODE2'] == 'sc') | (
                    df['VISCODE2'] == 'scmri'))]

        #if (row == '4858' and isImage and Modality == 'MRI'):
        #    print('%%%%%%%%%%%%%%%%%%%%%%%%%')
        #    print(sub_df)
        # print('xxxxxxxxxxxxxxxxxx',row)
        # print('子模块图像数量：',sub_df.shape[0])
        if sub_df.shape[0] > 0:
            bl_flag = sub_df['VISCODE2'].isin(['bl']).any()
            sc_flag = sub_df['VISCODE2'].isin(['sc', 'scmri']).any()
            m03_flage = sub_df['VISCODE2'].isin(['m03']).any()
            ##bl_flag=sub_df['RID'].isin(['bl']).any()
            ##sc_flag=sub_df['RID'].isin(['sc','scmri']).any()
            ##m03_flage=sub_df['RID'].isin(['m03']).any()

            if bl_flag and (sc_flag or m03_flage):  # bl 与其他两个阶段中的一个或两个 共存
                columns = sub_df.columns.values.tolist()
                mark = []
                for index_order, row in sub_df.iterrows():
                    visit = row['VISCODE2']
                    if visit == 'bl':
                        for i in range(0, len(columns)):
                            tmp_value = row[columns[i]]
                            if pd.isnull(tmp_value) or tmp_value == '-4' or tmp_value == '\"\"' or tmp_value == '':
                                mark.append([index_order, columns[i]])

                if not isImage:#不修补图像的基本信息
                    for i in range(0, len(mark)):
                        index = mark[i][0]
                        columns_name = mark[i][1]
                        for index_order, row in sub_df.iterrows():
                            visit = row['VISCODE2']
                            if visit != 'bl':
                                other_value = row[columns_name]
                                if not (pd.isnull(
                                        other_value) or other_value == '-4' or other_value == '\"\"' or other_value == ''):
                                    # print('修改前：',df.at[index, columns_name])
                                    df.at[index, columns_name] = other_value
                                    # print('修改后：',df.at[index, columns_name])
                                    break
            elif (not bl_flag) and (sc_flag and m03_flage):  # bl 不存在，sc和m03都存在
                columns = sub_df.columns.values.tolist()
                mark = []
                change_visitcoede = []
                for index_order, row in sub_df.iterrows():
                    visit = row['VISCODE2']
                    if visit == 'sc' or visit == 'scmri':
                        change_visitcoede.append(index_order)
                        for i in range(0, len(columns)):
                            tmp_value = row[columns[i]]
                            if pd.isnull(tmp_value) or tmp_value == '-4' or tmp_value == '\"\"' or tmp_value == '':
                                mark.append([index_order, columns[i]])

                if not isImage:#不修补图像的基本信息
                    for i in range(0, len(mark)):
                        index = mark[i][0]
                        columns_name = mark[i][1]
                        for index_order, row in sub_df.iterrows():
                            visit = row['VISCODE2']
                            if not (visit == 'sc' or visit == 'scmri'):
                                other_value = row[columns_name]
                                if not (pd.isnull(
                                        other_value) or other_value == '-4' or other_value == '\"\"' or other_value == ''):
                                    # print('修改前：',df.at[index, columns_name])
                                    df.at[index, columns_name] = other_value
                                    # print('修改后：',df.at[index, columns_name])
                                    break

                for index in change_visitcoede:
                    # print('修改前：',df.at[index_order, 'VISCODE2'])
                    df.at[index, 'VISCODE2'] = 'bl'
                    # print('修改后：',df.at[index_order, 'VISCODE2'])
            elif (not (bl_flag or m03_flage)) and sc_flag:  # 只有 sc 阶段存在
                for index_order, row in sub_df.iterrows():
                    # print('修改前：',df.at[index_order, 'VISCODE2'])
                    df.at[index_order, 'VISCODE2'] = 'bl'
                    # print('修改后：',df.at[index_order, 'VISCODE2'])
            elif (not (bl_flag or sc_flag)) and m03_flage:  # 只有m03阶段存在
                for index_order, row in sub_df.iterrows():
                    # print('修改前：',df.at[index_order, 'VISCODE2'])
                    df.at[index_order, 'VISCODE2'] = 'bl'
                    # print('修改后：',df.at[index_order, 'VISCODE2'])
        # print('xxxxxxxxxxxxxxxxxx',row)
        # print('xxxxxxxxxxxxxxxxxx',isImage)
        # print('xxxxxxxxxxxxxxxxxx',Modality)
        #if ((rRID == '4858') and isImage and (Modality == 'MRI')):
        #    print('%%%%%%%%%%%%%%%%%%%%%%%%%')
        #    sdf = df.loc[(df['RID'] == rRID) & (df['Modality'] == Modality)]
        #    print(sdf)
    return df

#df 缺失值填充
def set_missing_value(df):
    #df.where(df.notnull(),'-4')
    df =df.fillna('-4')
    df =df.where(df !='', '-4')
    df =df.where(df != '\"\"', '-4')
    return df


merge_df=pd.read_csv(merger_path, dtype=str)#综合信息
#merge_df.rename(columns={'VISCODE': 'VISCODE2'}, inplace=True)  # 为了统一表达 VISCODE 修改为VISCODE2
merge_df=combine_m03_and_sc_into_bl(merge_df)
merge_df=set_missing_value(merge_df)
merge_df=merge_df.set_index(['RID', 'VISCODE2'], drop=False)

image_df=pd.read_csv(image_path, dtype=str)#图像信息
image_df.rename(columns={'Visit': 'VISCODE2'}, inplace=True)  # 为了统一表达 Visit 修改为VISCODE2

#解决图像中的重复问题，否则随访时间长的用户将产生大量的数据。
image_df_MRI=image_df.loc[image_df['Modality'] == 'MRI']
image_df_MRI=image_df_MRI.sort_values(by=['RID', 'VISCODE2','Description'])
image_df_MRI = image_df_MRI.drop_duplicates(subset=['RID', 'VISCODE2','Sequence'])

image_df_PET=image_df.loc[image_df['Modality'] == 'PET']

image_df=image_df_MRI.append(image_df_PET)

image_df=combine_m03_and_sc_into_bl(image_df,isImage=True,Modality='MRI')
image_df=combine_m03_and_sc_into_bl(image_df,isImage=True,Modality='PET')
image_df=set_missing_value(image_df)

gen_dic={} #保存基因数据


def is_number(s):
    if (s=='-4') or s==("\"-4\"") or (s==-4):#-4为缺失值的填充
        return False
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False

#体液数据
csf_df=pd.read_csv(bio_csf, dtype=str)#脑脊液
csf_df=combine_m03_and_sc_into_bl(csf_df)
csf_df=set_missing_value(csf_df)
#csf_df=clear_bio_data(csf_df)

plasma_df=pd.read_csv(bio_plasma, dtype=str)#血浆，在这个版本里面包含了血液
plasma_df=combine_m03_and_sc_into_bl(plasma_df)
plasma_df=set_missing_value(plasma_df)
#plasma_df=clear_bio_data(plasma_df)

urine_df=pd.read_csv(bio_urine, dtype=str)#尿液，
urine_df=combine_m03_and_sc_into_bl(urine_df)
urine_df=set_missing_value(urine_df)
#urine_df=clear_bio_data(urine_df)

#量表数据
base_df=pd.read_csv(scale_base, dtype=str)
base_df=combine_m03_and_sc_into_bl(base_df)
base_df=set_missing_value(base_df)

cog_df=pd.read_csv(scale_cog, dtype=str)
cog_df=combine_m03_and_sc_into_bl(cog_df)
cog_df=set_missing_value(cog_df)

neuro_df=pd.read_csv(scale_neuro, dtype=str)
neuro_df=combine_m03_and_sc_into_bl(neuro_df)
neuro_df=set_missing_value(neuro_df)

cog_exame_df=pd.read_csv(scale_cog_exame, dtype=str)
cog_exame_df=combine_m03_and_sc_into_bl(cog_exame_df)
cog_exame_df=set_missing_value(cog_exame_df)

fb_df=pd.read_csv(scale_fb, dtype=str)
fb_df=combine_m03_and_sc_into_bl(fb_df)
fb_df=set_missing_value(fb_df)

exame_df=pd.read_csv(scale_exame, dtype=str)
exame_df=combine_m03_and_sc_into_bl(exame_df)
exame_df=set_missing_value(exame_df)

current_model={}

def select_model(pre_model,pooling):
    if pre_model+'_'+pooling in current_model:
        return current_model[pre_model+'_'+pooling]
    else:
        if pre_model == 'DenseNet201':  # 图片尺寸224*224，返回数据尺寸1*1920
            input_image = tf.keras.layers.Input([None, None, 3], dtype=tf.uint8)
            x = tf.cast(input_image, tf.float32)
            x = tf.keras.applications.densenet.preprocess_input(x)
            net = tf.keras.applications.DenseNet201(include_top=False, weights='imagenet', pooling=pooling)
            net.trainable = False
            x = net(x)
            model = tf.keras.Model(inputs=[input_image], outputs=[x])
            current_model[pre_model+'_'+pooling]=model
            return model
        else:
            return 0


#数据类型标签
#data_type=np.zeros(50)
#基础信息
#认知评价
#认识检查
#行为功能
#体格检查
#CCI
def get_scale(rid,viscode,scale_type):

    if scale_type == 'base':
        sub_scale_df = base_df.loc[(base_df['RID'] == rid) & (base_df['VISCODE2'] == viscode)]
    elif scale_type=='cog':
        sub_scale_df = cog_df.loc[(cog_df['RID'] == rid) & (cog_df['VISCODE2'] == viscode)]
    elif scale_type=='neuro':
        sub_scale_df = neuro_df.loc[(neuro_df['RID'] == rid) & (neuro_df['VISCODE2'] == viscode)]
    elif scale_type=='cog_exame':
        sub_scale_df = cog_exame_df.loc[(cog_exame_df['RID'] == rid) & (cog_exame_df['VISCODE2'] == viscode)]
    elif scale_type=='fb':
        sub_scale_df = fb_df.loc[(fb_df['RID'] == rid) & (fb_df['VISCODE2'] == viscode)]
    elif scale_type=='exame':
        sub_scale_df = exame_df.loc[(exame_df['RID'] == rid) & (exame_df['VISCODE2'] == viscode)]

    del sub_scale_df['RID']
    del sub_scale_df['VISCODE2']

    return sub_scale_df.values.tolist()

#血液
#尿液
#脑脊液
def get_bio(rid,viscode,bio_type):

    if bio_type=='csf':
        sub_bio_df = csf_df.loc[(csf_df['RID'] == rid) & (csf_df['VISCODE2'] == viscode)]
    elif bio_type=='plasma':
        sub_bio_df = plasma_df.loc[(plasma_df['RID'] == rid) & (plasma_df['VISCODE2'] == viscode)]
    elif bio_type=='urine':
        sub_bio_df = urine_df.loc[(urine_df['RID'] == rid) & (urine_df['VISCODE2'] == viscode)]

    del sub_bio_df['RID']
    del sub_bio_df['VISCODE2']

    return sub_bio_df.values.tolist()

#MRI
#PET-fda
#PET-av45

def convert_3Dto2DtoRGB(image_path,image_tmp_save_path,image_save_path):
    img = nib.load(image_path)  # 读取nii
    img_fdata = img.get_fdata()
    #fname = f.replace('.nii', '')  # 去掉nii的后缀名
    img_f_path = image_tmp_save_path
    # 创建nii对应的图像的文件夹
    if not os.path.exists(img_f_path):
        os.mkdir(img_f_path)  # 新建文件夹

    if not os.path.exists(image_save_path):
        os.mkdir(image_save_path)  # 新建文件夹

    # 开始转换为图像
    print(img.shape)
    (x, y, z) = img.shape
    for i in range(z):  # z是图像的序列
        silce = img_fdata[:, :, i]  # 选择哪个方向的切片都可以
        tmp_image_path=os.path.join(img_f_path, '{}.jpeg'.format(i))
        imageio.imwrite(tmp_image_path, silce)
        img = Image.open(tmp_image_path)
        img1 = img.convert('RGB')
        img1.save(os.path.join(image_save_path, '{}.jpeg'.format(i)))
        # 保存图像

def del_dir(path_file):
    ls = os.listdir(path_file)
    for i in ls:
        f_path = os.path.join(path_file, i)
        # 判断是否是一个目录,若是,则递归删除
        if os.path.isdir(f_path):
            del_dir(f_path)
        else:
            os.remove(f_path)
    os.rmdir(path_file)

def get_Image_data(image_type,rid,viscode,pre_model,start=-1,end=-1,sequence='-4',radiopharmaceutical='-4',pooling='avg'):
    #sub_image_df=image_df
    # 处理bl数据，如果存在bl，并且bl中有数据缺失的用筛查阶段的数据填补。如果没含有bl的，按照正常处理
    sub_image_df = image_df.loc[(image_df['RID'] == rid) & (image_df['VISCODE2'] == viscode) &
                                    (image_df['Modality'] == image_type) & (image_df['Sequence'] == sequence) &
                                    (image_df['Radiopharmaceutical'] == radiopharmaceutical)]

    retValue=[]
    if sub_image_df.shape[0]<1:
        print('此次就诊无影像资料： ',rid,'   ',viscode,'   ',image_type)

    for index_order, row in sub_image_df.iterrows():
        mri_type = [0]*10  # 0-1表示3T，2-3表示1.5T，4-5表示T1，6-7表示T2，8-9表示PD,全0表示非mri
        save_path = data_set_path+row['SavePath']
        image_id = row['Image ID']

        if not os.path.exists(save_path):#修正图像存放路径
            save_path=save_path.replace('.nii','')
            tmp_save_path_or=save_path+'.nii'
            if not os.path.exists(tmp_save_path_or):
                tmp_save_path_or=save_path+'a.nii'
                if not os.path.exists(tmp_save_path_or):
                    tmp_save_path_or = save_path + '_i00001.nii'
                    if not os.path.exists(tmp_save_path_or):
                        tmp_save_path_or = save_path + '_e1.nii'
                        if not os.path.exists(tmp_save_path_or):
                            #save_path = data_set_path + row['SavePath']#无法找到图像，还原原来的路径
                            #print(save_path)
                            if image_id in miss_image_dict:#手动补充的数据部分
                                save_path=data_set_path+miss_image_dict[image_id]
                            elif os.path.exists(data_set_path+'IMAGE/c_MRI/'+image_id+'.nii'):
                                save_path=data_set_path+'IMAGE/c_MRI/'+image_id+'.nii'
                            elif os.path.exists(data_set_path+'IMAGE/c_PET/'+image_id+'.nii'):
                                save_path = data_set_path + 'IMAGE/c_PET/' + image_id + '.nii'
                            else:
                                if image_id not in miss_imageID_set:
                                    print('RID:  ', rid, '   Viscode:   ', viscode, '    Study Date:',
                                          row['Study Date'], '    Modality: ', row['Modality'], '    ImageID: ',
                                          image_id, '    Radiopharmaceutical: ', row['Radiopharmaceutical'],
                                          '    不存在图片！！！！！！！')
                                    with open(miss_imageID_save_path, 'a') as f:
                                        f.write(rid + ',' + viscode + ',' + row['Study Date'] + ',' + row[
                                            'Modality'] + ',' + image_id + ',' + row['Radiopharmaceutical'] + '\n')
                                    miss_imageID_set.add(image_id)
                                return []
                        else:
                            save_path = tmp_save_path_or
                    else:
                        save_path = tmp_save_path_or
                else:
                    save_path=tmp_save_path_or
            else:
                save_path=tmp_save_path_or


        tmp_save_path = registration_save + image_id + '_grey'#存放配准之后的灰度图片，最后会被删除
        tmp_save_path_2 = registration_save + image_id + '_rgb'#存放配准之后的彩色图片
        tmp_array_save_path=image_array_path+pre_model+'_'+pooling+'/'+image_id+'.npy'#存放用神经网络抽取的图片特征
        array_path = os.path.dirname(tmp_array_save_path)
        if not os.path.exists(array_path):
            os.makedirs(array_path)

        field_strength = row['Field Strength']
        weighting = row['Weighting']

        if image_type == 'MRI':
            mri_type[4] = 1
            mri_type[5] = 1
            if weighting == 'T2':
                mri_type[4] = 0
                mri_type[5] = 0
                mri_type[6] = 1
                mri_type[7] = 1
            elif weighting == 'PD':
                mri_type[4] = 0
                mri_type[5] = 0
                mri_type[8] = 1
                mri_type[9] = 1
            mri_type[0]=1
            mri_type[1]=1
            if field_strength=='1.5':
                mri_type[0] = 0
                mri_type[1] = 0
                mri_type[2] = 1
                mri_type[3] = 1


        if not os.path.exists(tmp_save_path_2):#如果没有配准，则对图像进行配准
            registration_file_path = registration_save + image_id + '.nii'
            if image_type == 'MRI':
                moving = ants.image_read(save_path)
                tmp = moving.numpy()
                if tmp.ndim==4:
                    frames = int(tmp.shape[3])
                    slice_index = math.ceil(frames / 2.0)
                    tmp=tmp[:, :, :, slice_index - 1]
                moving = ants.from_numpy(tmp)
                fixed = fixed_T1
                if weighting == 'T2':
                    fixed = fixed_T2
                elif weighting == 'PD':
                    fixed = fixed_PD
                mytx = ants.registration(fixed=fixed, moving=moving, type_of_transform='SyN')
                # print(mytx)
                warped_moving = mytx['warpedmovout']
                ants.image_write(warped_moving, registration_file_path)
            elif image_type == 'PET':
                radiopharmaceutical = row['Radiopharmaceutical']
                moving = ants.image_read(save_path)
                tmp = moving.numpy()
                if tmp.ndim==4:
                    frames = int(tmp.shape[3])
                    slice_index = math.ceil(frames / 2.0)
                    tmp=tmp[:, :, :, slice_index - 1]
                moving = ants.from_numpy(tmp)
                fixed = fixed_FDG_PET
                if radiopharmaceutical == '18F-AV45':
                    fixed = fixed_AV45
                mytx = ants.registration(fixed=fixed, moving=moving, type_of_transform='SyN')
                # print(mytx)
                warped_moving = mytx['warpedmovout']
                ants.image_write(warped_moving, registration_file_path)

            convert_3Dto2DtoRGB(registration_file_path, tmp_save_path, tmp_save_path_2)
            os.remove(registration_file_path)
            del_dir(tmp_save_path)

        if not os.path.exists(tmp_array_save_path):#如果还没有利用模型抽取特征，则抽取特征
            image_count=len(os.listdir(tmp_save_path_2))

            if start==-1:
                start=0
            if end==-1:
                end=image_count
            image_array=[]
            for i in range(start,end):
                tmp_image_path=tmp_save_path_2+'/'+str(i)+'.jpeg'
                tmp_image = tf.image.decode_jpeg(tf.io.read_file(tmp_image_path))
                tmp_image = tf.image.resize(tmp_image, (224, 224), method='nearest')
                tmp_image = tf.reshape(tmp_image, [-1, 224, 224, 3])
                if i==0 or i==start:
                    image_array=tmp_image
                else:
                    image_array=tf.concat([image_array,tmp_image],axis=0)
            model=select_model(pre_model,pooling)
            #result=model(image_array).numpy()
            result = model.predict(image_array)
            print('Image size:   ',result.shape)#Image size:    (181, 1920)
            np.save(tmp_array_save_path,result)

        if pooling=='max':
            image_array_data = np.max(np.load(tmp_array_save_path), axis=0).tolist()
        elif pooling=='avg':
            image_array_data=np.mean(np.load(tmp_array_save_path),axis=0).tolist()
        #print('XXXXXXXXXXXX')
        #print(image_array_data)
        mri_type.extend(image_array_data)
        retValue.append(mri_type)
    return retValue

#基因
def read_gen(file_path):
    with open(file_path, 'r') as file_f:
        tmp_array = file_f.readlines()
        for i in range(1, len(tmp_array)):
            snp_array=tmp_array[i].rstrip('\n').strip().split(',')
            tmp_snp=[]
            for i in range(1,len(snp_array)):
                tmp_value=snp_array[i]
                if tmp_value=='':
                    tmp_value='-4'
                tmp_snp.append(tmp_value)
            if snp_array[0] not in gen_dic:
                gen_dic[snp_array[0]]=tmp_snp

#整理基因数据
def get_gen_dic():
    read_gen(gen_ANDI1GO2)
    read_gen(gen_ADNI3)
    read_gen(gen_ADNI1)
    read_gen(gen_ADNIGO2)

#判断是否存在数据
def is_null(data_list):
    flag=True
    for el in data_list:
        if not (pd.isnull(el) or (el=='-4') or (el=='\"\"') or (el=='')):
            flag=False
            break
    return flag

def is_null_for_2D(data_list):
    flag=True
    for lt in data_list:
        if not is_null(lt):
            flag=False
            break
    return flag

#列表拼接
def list_concat(list1,list2,l1=0,l2=0,total_len=-1):
    if len(list1) > 0:
        tmp_list1 = []
        for l in list1:
            if not is_null(l):
                tmp_list1.append(l)
        list1 = tmp_list1

    if len(list2) > 0:
        tmp_list2 = []
        for l in list2:
            if not is_null(l):
                tmp_list2.append(l)
        list2 = tmp_list2

    ret_value = []
    if len(list1)>0 and len(list2)<1:
        for l in list1:
            l=copy.copy(l)
            l.extend(['-4'] * l2)
            ret_value.append(l)
        #return ret_value
    elif len(list1)<1 and len(list2)>0:
        for l in list2:
            tmp=['-4']*l1
            tmp.extend(copy.copy(l))
            ret_value.append(tmp)
        #return ret_value
    elif len(list1)>0 and len(list2)>0:
        for le1 in list1:
            for le2 in list2:
                tmp_le1=copy.copy(le1)
                tmp_le1.extend(copy.copy(le2))
                ret_value.append(tmp_le1)

    if len(ret_value)>0 and total_len!=-1:
        for l in ret_value:
            if len(l)>0:
                plu_len = total_len - len(l)
                if plu_len > 0:
                    l.extend(['-4'] * plu_len)

    ret_value_2v=[]
    if len(ret_value) > 0:
        for l in ret_value:
            if not is_null(l):
                ret_value_2v.append(l)
    return ret_value_2v

def construct_merge_data(dt,list_len,merge_data):
    if len(merge_data)<1:
        return [['-4']*(sum(list_len)+1)]
    pre_list=['-4']*sum(list_len[0:dt])
    post_list=['-4']*sum(list_len[dt+1:len(list_len)])

    pre_list.extend(merge_data)
    pre_list.extend(post_list)
    pre_list.append('-4')

    return [pre_list]

#获取一次就诊中的一类数据
def get_one_step(dt,row_data,merge_data,bl_data,raw_data_len=2080,mri_type = [[0] * 10],is_image=False):
    #ret_str='Detail:   is_image: '+str(is_image)+'\n\n\n'
    #ret_str+='Initial Value:\n\n\n'
    #ret_str += 'Row data:\n'
    #ret_str += 'len:'+str(len(row_data))+'       '
    #for tmp_row_data in row_data:
    #    ret_str += str(len(tmp_row_data))+','
    #ret_str += '\n\n'
    #ret_str += 'Merge data:\n'
    #ret_str += 'len:' + str(len(merge_data)) + '       '
    #for tmp_merge_data in merge_data:
    #    ret_str += str(len(tmp_merge_data)) + ','
    #ret_str += '\n\n'
    #ret_str += 'BL data:\n'
    #ret_str += 'len:' + str(len(bl_data)) + '       '
    #for tmp_bl_data in bl_data:
    #    ret_str += str(len(tmp_bl_data)) + ','
    #ret_str += '\n\n\n'
    #ret_str += 'Change Value:\n\n\n'

    list_len=[6,8,7,0,15,0,0,0,7,1,1,1,3]
    #print('XXXXXXXXXXXXXXXXXX')
    #print(row_data)
    #print(merge_data)
    #print('')
    if not (is_null_for_2D(row_data) and is_null_for_2D(merge_data)):
        if not is_image:
            row_data = list_concat(mri_type, row_data)
            #ret_str += 'No Image row data:\n'
            #ret_str += 'len:' + str(len(row_data)) + '       '
            #for tmp_raw_data in row_data:
            #    ret_str += str(len(tmp_raw_data)) + ','
            #ret_str += '\n\n'


        data_type = [[0] * 50]
        data_type[0][2*dt], data_type[0][2*dt+1] = 1, 1

        #在适当的位置插入merge信息
        data_merge=construct_merge_data(dt,list_len,merge_data[0])
        #ret_str += 'Merge data:\n:\n'
        #ret_str += 'len:' + str(len(data_merge)) + '       '
        #for tmp_merge_data in data_merge:
        #    ret_str += str(len(tmp_merge_data)) + ','
        #ret_str += '\n\n'

        sample_data = list_concat(data_type, data_merge, 50, 50)
        #ret_str += 'Sum data 1:\n:\n'
        #ret_str += 'len:' + str(len(sample_data)) + '       '
        #for tmp_s_data in sample_data:
        #    ret_str += str(len(tmp_s_data)) + ','
        #ret_str += '\n\n'
        sample_data = list_concat(sample_data, bl_data, 100, 50)
        #ret_str += 'Sum data 2:\n:\n'
        #ret_str += 'len:' + str(len(sample_data)) + '       '
        #for tmp_s_data in sample_data:
        #   ret_str += str(len(tmp_s_data)) + ','
        #ret_str += '\n\n'
        sample_data = list_concat(sample_data, row_data, 150, total_len=raw_data_len)
        #ret_str += 'Sum data 1:\n:\n'
        #ret_str += 'len:' + str(len(sample_data)) + '       '
        #for tmp_s_data in sample_data:
        #    ret_str += str(len(tmp_s_data)) + ','
        #ret_str += '\n\n'
        return sample_data
    return []

#把每次就诊的数据拼接成列表的样子。
def generate_data(sample,extend,gaps):
    retValue=[]
    if len(sample)<1 and len(extend)>0:
        for el in extend:
            gaps_time=[gaps]*10
            gaps_time.extend(copy.copy(el))
            retValue.append([gaps_time])
    elif len(sample)>0 and len(extend)>0:
        for sel in sample:
            for eel in extend:
                gaps_time = [gaps] * 10
                gaps_time.extend(copy.copy(eel))
                tmp_sel=copy.copy(sel)
                tmp_sel.append(gaps_time)
                retValue.append(tmp_sel)
    return retValue

def generate_data_2v(data1,data2):

    if len(data1)>0 and len(data2)<1:
        return data1
    elif len(data1)<1 and len(data2)>0:
        return data2
    elif len(data1)>0 and len(data2)>0:
        retValue=[]
        for el1 in data1:
            for el2 in data2:
                tmp_el1=copy.copy(el1)
                tmp_el1.extend(copy.copy(el2))
                retValue.append(tmp_el1)
        return retValue
    return []

#将字符化为数值
def convert_2_number(list):
    if len(list)<1:
        return []
    else:
        retValue = []
        for i in range(len(list)):
            el = list[i]
            if el == 'Male':
                retValue.append('1')
            elif el == 'Female':
                retValue.append('2')
            elif el == 'Not Hisp/Latino':
                retValue.append('2')
            elif el == 'Hisp/Latino':
                retValue.append('1')
            elif el == 'Unknown' and i == 3:
                retValue.append('3')
            elif el == 'Married':
                retValue.append('1')
            elif el == 'Never married':
                retValue.append('4')
            elif el == 'Divorced':
                retValue.append('3')
            elif el == 'Widowed':
                retValue.append('2')
            elif el == 'Unknown' and i == 5:
                retValue.append('5')
            elif el == 'White':
                retValue.append('5')
            elif el == 'Asian':
                retValue.append('2')
            elif el == 'Black':
                retValue.append('4')
            elif el == 'More than one':
                retValue.append('6')
            elif el == 'Am Indian/Alaskan':
                retValue.append('1')
            elif el == 'Unknown' and i == 4:
                retValue.append('7')
            elif el == 'Hawaiian/Other PI':
                retValue.append('3')
            else:
                retValue.append(el)
        return retValue

#获取数据
def get_rowdata(methods,rid, viscode,raw_data_len):
    sample_data_dic={}#存放数据
    # 数据类型标签
    # 第0，1位基础信息，第2，3位认知信息，第4，5位认知检查，第6，7位精神信息，
    # 第8，9位行为功能检查，第10，11位体格检查,第12，13位血液检查，第14，15位尿液检查，
    # 第16，17位MRI检查，第18，19位18F-FDG PET, 第20，21位18F-AV45 PET，
    # 第22，23位基因检测，第24，25位脑脊液检测。
    #data_type = np.zeros(50)

    #返回的数据，每个样本的格式：1*2060。0-49数据类型标注，50-149重要数据，150-2079数据项
    #50年龄，51性别，52教育，53人种，54肤色，55婚姻状况，56apoe4基因，57fdg，58pib，59av45，
    #60-62脑脊液abeta，tau，ptau，63CDR-sb,64-67ADAS-11,13,Q4,68MMSE,69-75认知检查
    #immediate,learning forgetting,perc_forgetting,ldeltotal,digitscor,trabscor，76FAQ，
    #77MOCA，78-91EcogPtMem	EcogPtLang	EcogPtVisspat	EcogPtPlan	EcogPtOrgan	EcogPtDivatt	EcogPtTotal	EcogSPMem	EcogSPLang	EcogSPVisspat	EcogSPPlan	EcogSPOrgan	EcogSPDivatt	EcogSPTotal
    #92-97 核磁共振 Ventricles	Hippocampus	WholeBrain	Entorhinal	Fusiform	MidTemp	ICV
    #98-99 认知 mPACCdigit	mPACCtrailsB
    #基线信息 100-142 一共 43 个基线信息 CDRSB_bl	ADAS11_bl	ADAS13_bl	ADASQ4_bl	MMSE_bl	RAVLT_immediate_bl	RAVLT_learning_bl	RAVLT_forgetting_bl	RAVLT_perc_forgetting_bl	LDELTOTAL_BL	DIGITSCOR_bl	TRABSCOR_bl	FAQ_bl	mPACCdigit_bl	mPACCtrailsB_bl	Ventricles_bl	Hippocampus_bl	WholeBrain_bl	Entorhinal_bl	Fusiform_bl	MidTemp_bl	ICV_bl	MOCA_bl	EcogPtMem_bl	EcogPtLang_bl	EcogPtVisspat_bl	EcogPtPlan_bl	EcogPtOrgan_bl	EcogPtDivatt_bl	EcogPtTotal_bl	EcogSPMem_bl	EcogSPLang_bl	EcogSPVisspat_bl	EcogSPPlan_bl	EcogSPOrgan_bl	EcogSPDivatt_bl	EcogSPTotal_bl	ABETA_bl	TAU_bl	PTAU_bl	FDG_bl	PIB_bl	AV45_bl

    bl_data=merge_df.loc[(rid,viscode),['CDRSB_bl',	'ADAS11_bl', 'ADAS13_bl', 'ADASQ4_bl',
                                        'MMSE_bl', 'RAVLT_immediate_bl', 'RAVLT_learning_bl', 'RAVLT_forgetting_bl',
                                        'RAVLT_perc_forgetting_bl',	'LDELTOTAL_BL',	'DIGITSCOR_bl',	'TRABSCOR_bl',
                                        'FAQ_bl', 'mPACCdigit_bl', 'mPACCtrailsB_bl', 'Ventricles_bl',
                                        'Hippocampus_bl', 'WholeBrain_bl', 'Entorhinal_bl', 'Fusiform_bl',
                                        'MidTemp_bl', 'ICV_bl',	'MOCA_bl', 'EcogPtMem_bl',
                                        'EcogPtLang_bl', 'EcogPtVisspat_bl', 'EcogPtPlan_bl', 'EcogPtOrgan_bl',
                                        'EcogPtDivatt_bl', 'EcogPtTotal_bl', 'EcogSPMem_bl', 'EcogSPLang_bl',
                                        'EcogSPVisspat_bl',	'EcogSPPlan_bl', 'EcogSPOrgan_bl', 'EcogSPDivatt_bl',
                                        'EcogSPTotal_bl', 'ABETA_bl', 'TAU_bl',	'PTAU_bl', 'FDG_bl', 'AV45_bl']]
    bl_data=[bl_data.to_list()]
    print('基线数据的长度为： ',len(bl_data[0]))
    if len(bl_data)>0:
        bl_data[0].extend(['-4']*8)
    if viscode=='bl':#当前就诊为基线状态时，merge的基线信息不提供
        bl_data=[['4']*50]

    if methods[0]==1:
        mri_type = [[0] * 10]
        print('获取第一种数据 。。。 。。。')
        base_scale_data=get_scale(rid,viscode,'base')
        base_scale_data_merge = [merge_df.loc[(rid, viscode), ['AGE', 'PTGENDER', 'PTEDUCAT', 'PTETHCAT', 'PTRACCAT', 'PTMARRY']].to_list()]


        if not (is_null_for_2D(base_scale_data) and is_null_for_2D(base_scale_data_merge)):
            base_scale_data = list_concat(mri_type, base_scale_data)
            base_scale_data_merge = [convert_2_number(merge_df.loc[
                (rid, viscode), ['AGE', 'PTGENDER', 'PTEDUCAT', 'PTETHCAT', 'PTRACCAT', 'PTMARRY',
                                 'Years_bl']].to_list())]

            if len(base_scale_data_merge) > 0 and len(base_scale_data_merge[0]) > 0 and is_number(
                    base_scale_data_merge[0][0]) and is_number(base_scale_data_merge[0][-1]):
                base_scale_data_merge[0][0] = float(base_scale_data_merge[0][0]) + float(base_scale_data_merge[0][-1])

                del base_scale_data_merge[0][-1]

            data_type = [[0] * 50]
            data_type[0][0], data_type[0][1] = 1, 1

            base_scale_data_merge[0].extend(['-4'] * 44)

            sample_data = list_concat(data_type, base_scale_data_merge, 50, 50)
            sample_data = list_concat(sample_data, bl_data, 100, 50)
            sample_data = list_concat(sample_data, base_scale_data, 150, total_len=raw_data_len)
            print('获取第一种数据成功，数据长度为：  ',len(sample_data))
            if not is_null_for_2D(sample_data):
                sample_data_dic[0]=sample_data
                for tmp_data in sample_data:
                    tmp_len=len(tmp_data)
                    if tmp_len!=2080:
                        with open(data_len_save_path, 'a') as f:
                            f.write(rid+','+viscode+',0,'+str(tmp_len)+','+str(len(base_scale_data[0]))+','+str(len(base_scale_data_merge[0]))+','+str(len(bl_data[0]))+'\n')

    if methods[1]==1:
        print('获取第二种数据 。。。 。。。')
        cog_data=get_scale(rid,viscode,'cog')
        cog_merge_data=[merge_df.loc[(rid,viscode),['CDRSB','ADAS11','ADAS13','ADASQ4',
                                                    'MMSE','MOCA','mPACCdigit','mPACCtrailsB']].to_list()]
        sample_data=get_one_step(1,cog_data,cog_merge_data,bl_data)
        print('获取第二种数据成功，数据长度为：  ', len(sample_data))
        if not is_null_for_2D(sample_data):
            sample_data_dic[1]=sample_data
            for tmp_data in sample_data:
                tmp_len = len(tmp_data)
                if tmp_len != 2080:
                    with open(data_len_save_path, 'a') as f:
                        f.write(rid + ',' + viscode + ',1,' + str(tmp_len)+','+str(len(cog_data[0]))+','+str(len(cog_merge_data[0]))+','+str(len(bl_data[0])) + '\n')

    if methods[2]==1:
        print('获取第三种数据 。。。 。。。')
        cog_exame_data=get_scale(rid,viscode,'cog_exame')
        cog_exame_merge_data=[merge_df.loc[(rid,viscode),['RAVLT_immediate','RAVLT_learning','RAVLT_forgetting',
                                                          'RAVLT_perc_forgetting','LDELTOTAL','DIGITSCOR','TRABSCOR']].to_list()]
        sample_data=get_one_step(2,cog_exame_data,cog_exame_merge_data,bl_data)
        print('获取第三种数据成功，数据长度为：  ', len(sample_data))
        if not is_null_for_2D(sample_data):
            sample_data_dic[2]=sample_data
            for tmp_data in sample_data:
                tmp_len = len(tmp_data)
                if tmp_len != 2080:
                    with open(data_len_save_path, 'a') as f:
                        f.write(rid + ',' + viscode + ',2,' + str(tmp_len)+','+str(len(cog_exame_data[0]))+','+str(len(cog_exame_merge_data[0]))+','+str(len(bl_data[0])) + '\n')

    if methods[3]==1:
        print('获取第四种数据 。。。 。。。')
        neuro_data=get_scale(rid,viscode,'neuro')
        sample_data=get_one_step(3, neuro_data, [[]], bl_data)
        print('获取第四种数据成功，数据长度为：  ', len(sample_data))
        if not is_null_for_2D(sample_data):
            sample_data_dic[3] =sample_data
            for tmp_data in sample_data:
                tmp_len = len(tmp_data)
                if tmp_len != 2080:
                    with open(data_len_save_path, 'a') as f:
                        f.write(rid + ',' + viscode + ',3,' + str(tmp_len) +','+str(len(neuro_data[0]))+','+str(0)+','+str(len(bl_data[0]))+ '\n')

    if methods[4]==1:
        print('获取第五种数据 。。。 。。。')
        fb_data=get_scale(rid,viscode,'fb')
        fb_merge_data=[merge_df.loc[(rid, viscode), ['FAQ','EcogPtMem','EcogPtLang',
                                                     'EcogPtVisspat','EcogPtPlan','EcogPtOrgan',
                                                     'EcogPtDivatt','EcogPtTotal','EcogSPMem',
                                                     'EcogSPLang','EcogSPVisspat','EcogSPPlan',
                                                     'EcogSPOrgan', 'EcogSPDivatt', 'EcogSPTotal']].to_list()]
        sample_data=get_one_step(4, fb_data, fb_merge_data, bl_data)
        print('获取第五种数据成功，数据长度为：  ', len(sample_data))
        if not is_null_for_2D(sample_data):
            sample_data_dic[4] = sample_data
            for tmp_data in sample_data:
                tmp_len = len(tmp_data)
                if tmp_len != 2080:
                    with open(data_len_save_path, 'a') as f:
                        f.write(rid + ',' + viscode + ',4,' + str(tmp_len)+','+str(len(fb_data[0]))+','+str(len(fb_merge_data[0]))+','+str(len(bl_data[0])) + '\n')

    if methods[5]==1:
        print('获取第六种数据 。。。 。。。')
        exame_data=get_scale(rid,viscode,'exame')
        sample_data= get_one_step(5, exame_data, [[]], bl_data)
        print('获取第六种数据成功，数据长度为：  ', len(sample_data))
        if not is_null_for_2D(sample_data):
            sample_data_dic[5] =sample_data
            for tmp_data in sample_data:
                tmp_len = len(tmp_data)
                if tmp_len != 2080:
                    with open(data_len_save_path, 'a') as f:
                        f.write(rid + ',' + viscode + ',5,' + str(tmp_len)+','+str(len(exame_data[0]))+','+str(0)+','+str(len(bl_data[0])) + '\n')

    if methods[6]==1:
        print('获取第七种数据 。。。 。。。')
        plasm_data=get_bio(rid,viscode,'plasma')
        sample_data =get_one_step(6, plasm_data, [[]], bl_data)
        print('获取第七种数据成功，数据长度为：  ', len(sample_data))
        if not is_null_for_2D(sample_data):
            sample_data_dic[6] =sample_data
            for tmp_data in sample_data:
                tmp_len = len(tmp_data)
                if tmp_len != 2080:
                    with open(data_len_save_path, 'a') as f:
                        f.write(rid + ',' + viscode + ',6,' + str(tmp_len)+','+str(len(plasm_data[0]))+','+str(0)+','+str(len(bl_data[0])) + '\n')

    if methods[7]==1:
        print('获取第八种数据 。。。 。。。')
        urine_data=get_bio(rid,viscode,'urine')
        sample_data =get_one_step(7, urine_data, [[]], bl_data)
        print('获取第八种数据成功，数据长度为：  ', len(sample_data))
        if not is_null_for_2D(sample_data):
            sample_data_dic[7] =sample_data
            for tmp_data in sample_data:
                tmp_len = len(tmp_data)
                if tmp_len != 2080:
                    with open(data_len_save_path, 'a') as f:
                        f.write(rid + ',' + viscode + ',7,' + str(tmp_len)+','+str(len(urine_data[0]))+','+str(0)+','+str(len(bl_data[0])) + '\n')

    if methods[8]==1:
        print('获取第九种数据 。。。 。。。')
        mri_data=get_Image_data('MRI',rid,viscode,'DenseNet201',sequence='1')
        mri_merge_data=[merge_df.loc[(rid,viscode),['Ventricles',	'Hippocampus',	'WholeBrain',	'Entorhinal',	'Fusiform',	'MidTemp',	'ICV']].to_list()]
        sample_data =get_one_step(8, mri_data, mri_merge_data, bl_data,is_image=True)
        print('获取第九种数据成功，数据长度为：  ', len(sample_data))
        if not is_null_for_2D(sample_data):
            sample_data_dic[8] = sample_data
            for tmp_data in sample_data:
                tmp_len = len(tmp_data)
                if tmp_len != 2080:
                    with open(data_len_save_path, 'a') as f:
                        f.write(rid + ',' + viscode + ',8,' + str(tmp_len)+','+str(len(mri_data[0]))+','+str(len(mri_merge_data[0]))+','+str(len(bl_data[0])) + '\n')

    if methods[9]==1:
        print('获取第十种数据 。。。 。。。')
        fdg_data=get_Image_data('PET',rid,viscode,'DenseNet201',radiopharmaceutical='18F-FDG')
        fdg_merge_data=[[merge_df.loc[(rid,viscode),'FDG']]]
        sample_data=get_one_step(9,fdg_data,fdg_merge_data,bl_data,is_image=True)
        print('获取第十种数据成功，数据长度为：  ', len(sample_data))
        if not is_null_for_2D(sample_data):
            sample_data_dic[9] =sample_data
            for tmp_data in sample_data:
                tmp_len = len(tmp_data)
                if tmp_len != 2080:
                    with open(data_len_save_path, 'a') as f:
                        f.write(rid + ',' + viscode + ',9,' + str(tmp_len)+','+str(len(fdg_data[0]))+','+str(len(fdg_merge_data[0]))+','+str(len(bl_data[0])) + '\n')

    if methods[10]==1:
        print('获取第十一种数据 。。。 。。。')
        av45_data=get_Image_data('PET',rid,viscode,'DenseNet201',radiopharmaceutical='18F-AV45')
        av45_merge_data=[[merge_df.loc[(rid,viscode),'AV45']]]
        sample_data =get_one_step(10,av45_data,av45_merge_data,bl_data,is_image=True)
        print('获取第十一种数据成功，数据长度为：  ', len(sample_data))
        if not is_null_for_2D(sample_data):
            sample_data_dic[10] =sample_data
            for tmp_data in sample_data:
                tmp_len = len(tmp_data)
                if tmp_len != 2080:
                    with open(data_len_save_path, 'a') as f:
                        f.write(rid + ',' + viscode + ',10,' + str(tmp_len) +','+str(len(av45_data[0]))+','+str(len(av45_merge_data[0]))+','+str(len(bl_data[0]))+'\n')

    if methods[11]==1:
        print('获取第十二种数据 。。。 。。。')
        gen_data=[]
        if rid in gen_dic:
            gen_data=[gen_dic[rid]]
        gen_merge_data=[[merge_df.loc[(rid,viscode),'APOE4']]]
        sample_data=get_one_step(11,gen_data,gen_merge_data,bl_data)
        print('获取第十二种数据成功，数据长度为：  ', len(sample_data))
        if not is_null_for_2D(sample_data):
            sample_data_dic[11] = sample_data
            for tmp_data in sample_data:
                tmp_len = len(tmp_data)
                if tmp_len != 2080:
                    with open(data_len_save_path, 'a') as f:
                        f.write(rid + ',' + viscode + ',11,' + str(tmp_len) +','+str(len(gen_data[0]))+','+str(len(gen_merge_data[0]))+','+str(len(bl_data[0]))+ '\n')

    if methods[12]==1:
        print('获取第十三种数据 。。。 。。。')
        csf_data=get_bio(rid,viscode,'csf')
        csf_merge_data=[merge_df.loc[(rid,viscode),['ABETA',	'TAU',	'PTAU']].to_list()]
        sample_data = get_one_step(12,csf_data,csf_merge_data,bl_data)
        print('获取第十三种数据成功，数据长度为：  ', len(sample_data))
        if not is_null_for_2D(sample_data):
            sample_data_dic[12] = sample_data
            for tmp_data in sample_data:
                tmp_len = len(tmp_data)
                if tmp_len != 2080:
                    with open(data_len_save_path, 'a') as f:
                        f.write(rid + ',' + viscode + ',12,' + str(tmp_len)+','+str(len(csf_data[0]))+','+str(len(csf_merge_data[0]))+','+str(len(bl_data[0])) + '\n')

    return sample_data_dic

#按照限制条件获取数列
def get_list(method_num):
    retValue=[]
    for i in range(1,2**method_num):
        bin_i=str(bin(i))
        bin_i=bin_i[2:len(bin_i)]#前面有两个标示2进制的符号，所以要去掉
        bin_i=''.join(reversed(bin_i))
        i_len=len(bin_i)
        bin_i=bin_i+'0'*(method_num-i_len)

        if bin_i[0]=='1':
            if not (bin_i[8]=='1' and (bin_i[0]=='0' or bin_i[1]=='0' or bin_i[2]=='0' or bin_i[3]=='0' or bin_i[4]=='0' or bin_i[5]=='0')):
                if not ((bin_i[9]=='1' or bin_i[10]=='1') and (bin_i[0]=='0' or bin_i[1]=='0' or bin_i[2]=='0' or bin_i[3]=='0' or bin_i[4]=='0' or bin_i[5]=='0' or bin_i[6]=='0' or bin_i[7]=='0' or bin_i[8]=='0')):
                    if not (bin_i[11]=='1' and (bin_i[0]=='0' or bin_i[1]=='0' or bin_i[2]=='0' or bin_i[3]=='0' or bin_i[4]=='0' or bin_i[5]=='0' or bin_i[6]=='0' or bin_i[7]=='0' or bin_i[8]=='0')):
                        if not (bin_i[12]=='1' and (bin_i[0]=='0' or bin_i[1]=='0' or bin_i[2]=='0' or bin_i[3]=='0' or bin_i[4]=='0' or bin_i[5]=='0' or bin_i[6]=='0' or bin_i[7]=='0' or bin_i[8]=='0' or bin_i[9]=='0' or bin_i[10]=='0' or bin_i[11]=='0')):
                            retValue.append(bin_i)
    return retValue

def convert_str2int(list):
    retValue=[]
    for el in list:
        tmpValue=[]
        for elel in el:
            tmpValue.append(int(elel))
        retValue.append(tmpValue)
    return retValue

#生成当前就诊的就诊路径组合
def generate_list(list_all,viscode,data_dic):
    list_set=set()
    for lel in list_all:
        tmp_lel=copy.copy(lel)#复制之后不会改变原来的数组
        if not(viscode!='bl' and tmp_lel[11]=='1'):
            for i in range(0, len(tmp_lel)):
                index_value=tmp_lel[i]
                if index_value=='1' and (i not in data_dic):
                    lel1=tmp_lel[:i]
                    lel2=tmp_lel[i+1:]
                    tmp_lel=lel1+'0'+lel2
            list_set.add(tmp_lel)

    final_list=list(list_set)
    final_list=convert_str2int(final_list)
    print('当前就诊产生的组合数目为：    ',len(final_list))

    retValue=[]
    for el in final_list:
        sample_dataset = []
        for i in range(0,len(el)):
            if el[i]==1:
                tmp_one_step_data=data_dic[i]
                print('current   待拼接数据长度为：    ', len(tmp_one_step_data))
                sample_dataset=generate_data(sample_dataset,tmp_one_step_data,0)
                print('current   拼接完成的数据长度为：    ', len(sample_dataset))

        retValue.extend(sample_dataset)
    return retValue

def get_examination_data(visit):

    sample_dataset=[]
    history_data=[]
    gaps=[]
    visitcodes=[]

    num_list=get_list(13)
    for i in range(0,len(visit)):
        rid=visit[i][0]
        viscode=visit[i][1]
        gap_time=visit[i][2]

        methods_one_hot=[1]*13

        if not (viscode=='bl'):#不是在基线情况下，不做基因检查
            methods_one_hot[11]=0
        print('获取一次数字字典： ',rid,'   ',viscode,'   ',methods_one_hot)
        sample_dic=get_rowdata(methods_one_hot,rid,viscode,2080)
        print('一次数字字典获取完毕，长度为：   ',len(sample_dic))
        if len(sample_dic)>0:
            gaps.append(gap_time)
            visitcodes.append(viscode)
            history_data.append(sample_dic)

    if len(history_data)>0:
        if len(history_data) > 1:
            for i in range(0, len(history_data) - 1):
                print('数据待拼接位置为：    ', i)
                if len(history_data[i]) > 0:
                    for j in range(0, 13):
                        if j in history_data[i]:
                            tmp_data = history_data[i][j]
                            print('待拼接数据长度为：    ',len(tmp_data))
                            sample_dataset = generate_data(sample_dataset,tmp_data,gaps[i])
                            print('拼接完成的数据长度为：    ', len(sample_dataset))


        last_visit_data=generate_list(num_list,visitcodes[-1],history_data[-1])

        final_dataset=generate_data_2v(sample_dataset,last_visit_data)
        return final_dataset
    return []

# 数据集生成
##
def create_dataset(smc_flag=0, mci2ad=36):#smc_flag=0表示如果基线的时候是smc并且诊断是cn才算smc，非基线时候的smc不考虑
    #diagnosis_table = fix_diagnosis()
    diagnosis_table = pd.read_csv(merger_path, dtype=str)
    diagnosis_table.rename(columns={'VISCODE': 'VISCODE2'}, inplace=True)  # 为了统一表达 VISCODE 修改为VISCODE2
    sub_diagnosis_table = diagnosis_table[['RID', 'VISCODE2', 'EXAMDATE', 'DX', 'DX_bl']]
    sub_diagnosis_table = sub_diagnosis_table.sort_values(by=['RID', 'EXAMDATE'])
    sub_diagnosis_table = sub_diagnosis_table.dropna(axis=0, how='any')  # 删除诊断DX为空的行。<这些行代表随访时间没有到，由于其他原因收集的患者信息>

    cAD = []  # 目前处于AD状态
    cMCI = []  # 目前处于MCI状态
    cCN = []  # 目前处于正常人状态

    cSMC = []  # 目前处于重大记忆问题状态

    sMCI = []  # 目前处于MCI状态，未来一段时间内不会转变为AD
    pMCI = []  # 目前处于MCI状态，未来一段时间内会转变为AD

    rID_set = sub_diagnosis_table['RID'].drop_duplicates()

    for index, row in rID_set.items():
        rRID = row
        sub_diagnosis = sub_diagnosis_table.loc[sub_diagnosis_table['RID'] == rRID]

        tmp_viscode = []  # 访问码，表示离基线多久
        tmp_diagnosis = []  # 当前访问码下的诊断
        tmp_diagnosis_bl = []  # 在基线时候的诊断。由于经过修改，在基线时不是AD的病人这个值会随着当前诊断而改变。不再表示基线时的诊断

        for sub_index, sub_row in sub_diagnosis.iterrows():
            tmp_viscode.append(sub_row['VISCODE2'])
            tmp_diagnosis.append(sub_row['DX'])
            tmp_diagnosis_bl.append(sub_row['DX_bl'])
        for i in range(0, len(tmp_viscode)):
            rViscode = tmp_viscode[i]
            rDXbl = tmp_diagnosis_bl[i]
            rDX = tmp_diagnosis[i]

            r_time = 0
            if rViscode != 'bl':
                r_time = int(rViscode.replace("m", ""))

            tmp_sample = []
            for index in range(0, i):  # 添加历史就诊记录
                t_time = 0
                vCode = tmp_viscode[index]
                if vCode != 'bl':
                    # print('%%%%%%%%%%%%%')
                    # print(vCode.replace("m",""))
                    t_time = int(vCode.replace("m", ""))
                s_tmp_sample = [rRID, vCode, r_time - t_time]
                tmp_sample.append(s_tmp_sample)

            tmp_sample.append([rRID, rViscode, 0])  # 添加当前就诊记录

            if rDXbl == 'SMC' and rDX == 'CN':
                if smc_flag == 0:  # 只取基线时候的数据
                    if rViscode == 'bl':
                        cSMC.append(tmp_sample)
                elif smc_flag == 1:
                    cSMC.append(tmp_sample)
            else:
                if rDX == 'Dementia':
                    cAD.append(tmp_sample)
                elif rDX == 'CN' and rDXbl != 'SMC':
                    cCN.append(tmp_sample)
                elif rDX == 'MCI':
                    cMCI.append(tmp_sample)

                    f_sub_diagnosis = []
                    for sub_index in range(i + 1, len(tmp_diagnosis)):
                        f_sub_diagnosis.append(tmp_diagnosis[sub_index])

                    if 'Dementia' in f_sub_diagnosis:
                        for sub_index in range(i + 1, len(tmp_diagnosis)):
                            if tmp_diagnosis[sub_index] == 'Dementia':
                                t_ad_time = 0
                                if tmp_viscode[sub_index] != 'bl':
                                    t_ad_time = int(tmp_viscode[sub_index].replace("m", ""))
                                if t_ad_time - r_time <= mci2ad:
                                    pMCI.append(tmp_sample)
                                else:
                                    sMCI.append(tmp_sample)
                                break
                    else:
                        last_index = len(tmp_diagnosis) - 1
                        t_last_time = 0
                        if tmp_viscode[last_index] != 'bl':
                            t_last_time = int(tmp_viscode[last_index].replace("m", ""))
                        if t_last_time - r_time >= mci2ad:
                            sMCI.append(tmp_sample)
    print('AD病人样本总数量： ' + str(len(cAD)))
    print('正常人总样本量：' + str(len(cCN)))
    print('轻度认知功能障碍患者样本总数量：' + str(len(cMCI)))
    print('稳定的轻度功能障碍患者样本总数量：' + str(len(sMCI)))
    print('进展型轻度功能障碍患者样本总数量：' + str(len(pMCI)))
    print('重大记忆力问题患者样本总数量：' + str(len(cSMC)))
    #print(cAD)
    return [cAD, cCN, cMCI, sMCI, pMCI, cSMC]


# 生成字符串型的属性
def bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))

def tf_writ(d_x,d_y,writer):
    sample_x = list(map(lambda x: np.float32(x), d_x))
    sample_x_np = np.array(sample_x).reshape(1, len(d_x))
    sample_x_np = sample_x_np.tobytes()

    sample_y = list(map(lambda x: np.float32(x), d_y))
    sample_y_np = np.array(sample_y).reshape(1, len(d_y))
    sample_y_np = sample_y_np.tobytes()


    # sample_x_tb = sample_x_np.tobytes()
    # sample_y_tb = sample_y_np.tobytes()

    example = tf.train.Example(features=tf.train.Features(feature={
        "label": bytes_feature(sample_y_np),
        # 使得patient对应的label为1， normal 对应的label为0
        "data": bytes_feature(sample_x_np),
    }))
    writer.write(example.SerializeToString())  # 序列化为字符串


def tf_writ_acm(d_x,d_ad_y,d_cn_y,d_mci_y,rid,viscode,writer):
    sample_x = list(map(lambda x: np.float32(x), d_x))
    sample_x_np = np.array(sample_x)
    sample_x_np = sample_x_np.tobytes()

    sample_ad_y = list(map(lambda x: np.float32(x), d_ad_y))
    sample_ad_y_np = np.array(sample_ad_y)
    sample_ad_y_np = sample_ad_y_np.tobytes()

    sample_cn_y = list(map(lambda x: np.float32(x), d_cn_y))
    sample_cn_y_np = np.array(sample_cn_y)
    sample_cn_y_np = sample_cn_y_np.tobytes()

    sample_mci_y = list(map(lambda x: np.float32(x), d_mci_y))
    sample_mci_y_np = np.array(sample_mci_y)
    sample_mci_y_np = sample_mci_y_np.tobytes()

    sample_rid_np=rid.encode('utf-8')
    sample_viscode_np=viscode.encode('utf-8')
    #sample_rid_np = list(map(lambda x: x.encode('utf-8'), rid))
    #sample_rid_np = np.array(sample_rid)
    #sample_rid_np = sample_rid_np.tobytes()

    #sample_viscode_np = list(map(lambda x: x.encode('utf-8'), viscode))
    #sample_viscode_np = np.array(viscode)
    #sample_viscode_np = sample_viscode_np.tobytes()

    #sample_methond = list(map(lambda x: np.float32(x), methond))
    #sample_methond_np = np.array(sample_methond)
    #sample_methond_np = sample_methond_np.tobytes()

    # sample_x_tb = sample_x_np.tobytes()
    # sample_y_tb = sample_y_np.tobytes()

    example = tf.train.Example(features=tf.train.Features(feature={
        "label_ad": bytes_feature(sample_ad_y_np),
        "label_cn": bytes_feature(sample_cn_y_np),
        "label_mci": bytes_feature(sample_mci_y_np),
        "rid": bytes_feature(sample_rid_np),
        "viscode": bytes_feature(sample_viscode_np),
        #"methond": bytes_feature(sample_methond_np),
        # 使得patient对应的label为1， normal 对应的label为0
        "data": bytes_feature(sample_x_np),
    }))
    writer.write(example.SerializeToString())  # 序列化为字符串

def tf_writ_sp(d_x,d_s_y,d_p_y,rid,viscode,writer):
    sample_x = list(map(lambda x: np.float32(x), d_x))
    sample_x_np = np.array(sample_x)
    sample_x_np = sample_x_np.tobytes()

    sample_s_y = list(map(lambda x: np.float32(x), d_s_y))
    sample_s_y_np = np.array(sample_s_y)
    sample_s_y_np = sample_s_y_np.tobytes()

    sample_p_y = list(map(lambda x: np.float32(x), d_p_y))
    sample_p_y_np = np.array(sample_p_y)
    sample_p_y_np = sample_p_y_np.tobytes()

    sample_rid_np = rid.encode('utf-8')
    sample_viscode_np = viscode.encode('utf-8')

    #sample_rid = list(map(lambda x: np.float32(x), rid))
    #sample_rid_np = np.array(rid)
    #sample_rid_np = sample_rid_np.tobytes()

    #sample_viscode_np = np.array(viscode)
    #sample_viscode_np = sample_viscode_np.tobytes()

    #sample_methond = list(map(lambda x: np.float32(x), methond))
    #sample_methond_np = np.array(sample_methond)
    #sample_methond_np = sample_methond_np.tobytes()


    # sample_x_tb = sample_x_np.tobytes()
    # sample_y_tb = sample_y_np.tobytes()

    example = tf.train.Example(features=tf.train.Features(feature={
        "label_s": bytes_feature(sample_s_y_np),
        "label_p": bytes_feature(sample_p_y_np),
        "rid": bytes_feature(sample_rid_np),
        "viscode": bytes_feature(sample_viscode_np),
        #"methond": bytes_feature(sample_methond_np),
        # 使得patient对应的label为1， normal 对应的label为0
        "data": bytes_feature(sample_x_np),
    }))
    writer.write(example.SerializeToString())  # 序列化为字符串


#展开样本,形成一维的样本
def convert_null_2_value(sample):
    retValue=[]
    for tmp1 in sample:
        for tmp2 in tmp1:
            if pd.isnull(tmp2) or tmp2=='':
                retValue.append('-4')
            else:
                retValue.append(tmp2)
    return retValue

def get_list_str(list):
    retValue=''
    for el in list:
        retValue+=str(el)+'\n'
    retValue+='\n'
    return retValue

def create_tensor_dataset():
    #cAD,cCN,cMCI,sMCI,pMCI,cSMC=create_dataset()
    sample_list=create_dataset()
    get_gen_dic()#加载基因

    train_rid=set()
    eval_rid=set()
    test_rid=set()

    train_sp_rid=set()
    eval_sp_rid=set()
    test_sp_rid=set()

    train_ac_rid = set()
    eval_ac_rid = set()
    test_ac_rid = set()

    #训练集80%，校验集5%，测试集15%

    #AD vs CN vs MCI

    #用于存放数据的下标，并打乱了顺序
    task_acm_train_index_set=[]
    task_acm_eval_index_set=[]
    task_acm_test_index_set=[]



    #sMCI vs pMCI

    # 用于存放数据的下标，并打乱了顺序
    task_sp_train_index_set=[]
    task_sp_eval_index_set=[]
    task_sp_test_index_set=[]

    #用于存放数据的下标
    task_smc_test_set=[]

    # 用于存放数据的下标，并打乱了顺序
    task_ac_train_index_set = []
    task_ac_eval_index_set = []
    task_ac_test_index_set = []

    # 用于存放数据的下标
    task_mci_test_set = []

    #获取下标
    for i in range(5, len(sample_list)):
        dataset = sample_list[i]
        if i < 3:
            for j in range(0,len(dataset)):
                p_v=dataset[j]
                rid = p_v[0][0]
                if rid in train_rid:
                    task_acm_train_index_set.append([i,j])
                elif rid in eval_rid:
                    task_acm_eval_index_set.append([i,j])
                elif rid in test_rid:
                    task_acm_test_index_set.append([i,j])
                else:
                    rate = random.random()
                    if rate < 0.8:
                        task_acm_train_index_set.append([i,j])
                        train_rid.add(rid)
                    elif rate >= 0.8 and rate < 0.85:
                        task_acm_eval_index_set.append([i,j])
                        eval_rid.add(rid)
                    elif rate >= 0.85:
                        task_acm_test_index_set.append([i,j])
                        test_rid.add(rid)
        elif i >= 3 and i < 5:
            for j in range(0,len(dataset)):  # p_v_mci 格式 <rid,viscode,gap>
                p_v_mci=dataset[j]
                sp_rid = p_v_mci[0][0]
                if sp_rid in train_sp_rid:
                    task_sp_train_index_set.append([i,j])
                elif sp_rid in eval_sp_rid:
                    task_sp_eval_index_set.append([i,j])
                elif sp_rid in test_sp_rid:
                    task_sp_test_index_set.append([i,j])
                else:
                    sp_rate = random.random()
                    if sp_rate < 0.8:
                        task_sp_train_index_set.append([i,j])
                        train_sp_rid.add(sp_rid)
                    elif sp_rate >= 0.8 and sp_rate < 0.85:
                        task_sp_eval_index_set.append([i,j])
                        eval_sp_rid.add(sp_rid)
                    elif sp_rate >= 0.85:
                        task_sp_test_index_set.append([i,j])
                        test_sp_rid.add(sp_rid)
        elif i==5:
            for j in range(0,len(dataset)):  # p_v_mci 格式 <rid,viscode,gap>
                task_smc_test_set.append([i,j])

    #打乱顺序
    random.shuffle(task_acm_train_index_set)
    random.shuffle(task_acm_eval_index_set)
    random.shuffle(task_acm_test_index_set)

    random.shuffle(task_sp_train_index_set)
    random.shuffle(task_sp_eval_index_set)
    random.shuffle(task_sp_test_index_set)


    # 创建一个writer来写TFRecord文件
    tfrecord_save_path='/data/huangyunyou/TFRecord_2v/'
    writer_acm_train = tf.io.TFRecordWriter(tfrecord_save_path+'acm_train.tfrecord')
    writer_acm_valid = tf.io.TFRecordWriter(tfrecord_save_path+'acm_eval.tfrecord')
    writer_acm_test = tf.io.TFRecordWriter(tfrecord_save_path+'acm_test.tfrecord')


    writer_sp_train = tf.io.TFRecordWriter(tfrecord_save_path+'sp_train.tfrecord')
    writer_sp_valid = tf.io.TFRecordWriter(tfrecord_save_path+'sp_eval.tfrecord')
    writer_sp_test = tf.io.TFRecordWriter(tfrecord_save_path+'sp_test.tfrecord')

    writer_smc_test = tf.io.TFRecordWriter(tfrecord_save_path + 'smc_test.tfrecord')

    writer_ac_train = tf.io.TFRecordWriter(tfrecord_save_path + 'ac_train.tfrecord')
    writer_ac_valid = tf.io.TFRecordWriter(tfrecord_save_path + 'ac_eval.tfrecord')
    writer_ac_test = tf.io.TFRecordWriter(tfrecord_save_path + 'ac_test.tfrecord')

    writer_mci_test = tf.io.TFRecordWriter(tfrecord_save_path + 'mci_test.tfrecord')


    sample_count=0
    max_sample_len=0
    acm_tr_count=0
    acm_va_count=0
    acm_te_count=0
    sp_tr_count=0
    sp_va_count=0
    sp_te_count=0

    ad_count = 0
    cn_count = 0
    mci_count = 0
    smci_count = 0
    pmci_count = 0
    smc_te_count=0

    for k in range(6,7):
        task_dataset=task_acm_train_index_set
        if k==1:
           task_dataset=task_acm_eval_index_set
        elif k==2:
            task_dataset=task_acm_test_index_set
        elif k==3:
            task_dataset=task_sp_train_index_set
        elif k==4:
            task_dataset=task_sp_eval_index_set
        elif k==5:
            task_dataset=task_sp_test_index_set
        elif k==6:
            task_dataset=task_smc_test_set

        for index in task_dataset:
            i = index[0]
            j = index[1]
            label_ad = 0
            label_cn = 0
            label_mci = 0
            label_sMCI = 0
            label_pMCI = 0

            if i < 3:
                label_ad = 1
                label_cn = 1
                label_mci = 1
                if i == 0:
                    label_cn = 0
                    label_mci = 0
                elif i == 1:
                    label_ad = 0
                    label_mci = 0
                elif i == 2:
                    label_ad = 0
                    label_cn = 0
            elif i >= 3 and i < 5:
                label_sMCI = 1
                label_pMCI = 1
                if i == 3:
                    label_pMCI = 0
                elif i == 4:
                    label_sMCI = 0

            sample_with_viscode = sample_list[i][j]
            rid = sample_with_viscode[-1][0]
            viscode = sample_with_viscode[-1][1]
            print('就诊记录： ',sample_with_viscode)
            sample_data = get_examination_data(sample_with_viscode)  # 获取数据
            print('就诊记录条数： ', sample_with_viscode, '    ',len(sample_data))
            if len(sample_data)<350:
                # sample_count=len(sample_data)
                for tmp_sample in sample_data:

                    if label_ad==1:
                        ad_count+=1
                    elif label_cn==1:
                        cn_count+=1
                    elif label_mci==1:
                        mci_count+=1
                    elif label_sMCI==1:
                        smci_count+=1
                    elif label_pMCI==1:
                        pmci_count+=1

                    tmp_sample=sum(tmp_sample, [])
                    tmp_len = len(tmp_sample)
                    if tmp_len%2090 != 0:
                        with open(data_len_save_path_2v, 'a') as f:
                            f.write(rid + ',' + viscode + ',' + str(tmp_len) + '\n')

                    sample_count += 1
                    len_sample = len(tmp_sample)
                    if len_sample > max_sample_len:
                        max_sample_len = len_sample
                    #if sample_count < 200:
                    #    with open('/data/huangyunyou/test/sample.txt', 'a') as f:
                    #        f.write(get_list_str(tmp_sample))
                    # print(tmp_sample)
                    print(rid)
                    print('*****************')
                    if k == 0:
                        acm_tr_count += 1
                        tf_writ_acm(tmp_sample, [label_ad], [label_cn], [label_mci], rid, viscode,
                                    writer_acm_train)
                        #if sample_count < 200:
                        #    tmp_str = str(label_ad) + '\n' + str(label_cn) + '\n' + str(
                        #        label_mci) + '\n' + rid + '\n' + viscode + '\n' + str(tmp_sample[-1][10:60]) + '\n\n'
                        #    with open('/data/huangyunyou/test/sample.txt', 'a') as f:
                        #        f.write(tmp_str)
                    elif k == 1:
                        acm_va_count += 1
                        tf_writ_acm(tmp_sample, [label_ad], [label_cn], [label_mci], rid, viscode,
                                    writer_acm_valid)
                    elif k == 2:
                        acm_te_count += 1
                        tf_writ_acm(tmp_sample, [label_ad], [label_cn], [label_mci], rid, viscode,
                                    writer_acm_test)
                    elif k == 3:
                        sp_tr_count += 1
                        tf_writ_sp(tmp_sample, [label_sMCI], [label_pMCI], rid, viscode, writer_sp_train)
                    elif k == 4:
                        sp_va_count += 1
                        tf_writ_sp(tmp_sample, [label_sMCI], [label_pMCI], rid, viscode, writer_sp_valid)
                    elif k == 5:
                        sp_te_count += 1
                        tf_writ_sp(tmp_sample, [label_sMCI], [label_pMCI], rid, viscode, writer_sp_test)
                    elif k == 6:
                        smc_te_count += 1
                        tf_writ_acm(tmp_sample, [0], [0],[0], rid, viscode, writer_smc_test)
            else:
                print('！！！！！！！！！！！就诊记录条数过多，不写入文件： ', sample_with_viscode, '    ', len(sample_data))
                with open(abnormal_number_viscode_save_path, 'a') as f:
                    f.write(str(sample_with_viscode)+ '\n')

    print('##################')
    print('##################')
    print('##################')
    print('数据的最大长度为： ', max_sample_len)
    print('数据总样本数： ', sample_count)
    print('acm训练样本数： ', acm_tr_count)
    print('acm校验样本数： ', acm_va_count)
    print('acm测试样本数： ', acm_te_count)
    print('sp训练样本数： ', sp_tr_count)
    print('sp校验样本数： ', sp_va_count)
    print('sp测试样本数： ', sp_te_count)
    print('ad样本数： ', ad_count)
    print('cn样本数： ', cn_count)
    print('mci样本数： ', mci_count)
    print('smci样本数： ', smci_count)
    print('pmci样本数： ', pmci_count)
    print('smc样本数： ', smc_te_count)
    print('##################')
    print('##################')
    print('##################')
    with open(sample_statistics_information_save_path, 'a') as f:
        f.write('数据的最大长度为： ' + str(max_sample_len) + '\n数据总样本数： ' + str(sample_count) +
                '\n acm训练样本数： ' + str(acm_tr_count) + '\n acm校验样本数： ' + str(acm_va_count) +
                '\n acm测试样本数： ' + str(acm_te_count) + '\n sp训练样本数： ' + str(sp_tr_count) +
                '\n sp校验样本数： ' + str(sp_va_count) + '\n sp测试样本数： ' + str(sp_te_count) +
                '\n ad样本数： ' + str(ad_count) + '\n cn样本数： ' + str(cn_count) +
                '\n mci样本数： ' + str(mci_count) + '\n smci测试样本数： ' + str(smci_count) +
                '\n pmci校验样本数： ' + str(pmci_count)
                )


create_tensor_dataset()


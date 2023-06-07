#-*- coding:utf-8 -*-
import csv
import sys
import os
import pandas as pd
import re
import datetime

medhist='/mnt/sdb/pwb/StudyData/MedicalHistory/MedicalHistory/'
medhist_1go2=medhist+'MEDHIST.csv'
medhist_3=medhist+'INITHEALTH.csv'
recbllog=medhist+'RECBLLOG.csv'

exams='/mnt/sdb/pwb/StudyData/MedicalHistory/PhysicalNeurologicalExams/'
neuroexm=exams+'NEUROEXM.csv'
physical=exams+'PHYSICAL.csv'
vials=exams+'VITALS.csv'
blscheck=exams+'BLSCHECK.csv'


subject_demographics='/mnt/sdb/pwb/StudyData/SubjectCharacteristics/SubjectDemographics/PTDEMOG.csv'
family_history='/mnt/sdb/pwb/StudyData/SubjectCharacteristics/FamilyHistory/'
par=family_history+'FAMXHPAR.csv'
sib=family_history+'FAMXHSIB.csv'
related=family_history+'FHQ.csv'
re_related=family_history+'RECFHQ.csv'

adsxlist='/mnt/sdb/pwb/StudyData/Assessments/Diagnosis/ADSXLIST.csv'

viscode_2_viscode2_dic_path='/mnt/sdb/pwb/StudyData/Enrollment/Enrollment/ADNI2_VISITID.csv'

save_path='/mnt/sdb/pwb/scale/'


Neur_path='/mnt/sdb/pwb/StudyData/Assessments/Neuropsychological/'
mmse=Neur_path+'MMSE.csv'
moca=Neur_path+'MOCA.csv'
neurobat=Neur_path+'NEUROBAT.csv'
cci=Neur_path+'CCI.csv'
cdr=Neur_path+'CDR.csv'
npi=Neur_path+'NPI.csv'
npiq=Neur_path+'NPIQ.csv'
gds=Neur_path+'GDSCALE.csv'
ecogpt=Neur_path+'ECOGPT.csv'
ecogsp=Neur_path+'ECOGSP.csv'
faq=Neur_path+'FAQ.csv'

#按照时间来修改错误的viscode。
#本次修改只考虑年度信息，因为随着病人的随访时间增加，
# 随访的时间变得不稳定，只要在不同年度就可以认为是两个不同的viscode。
def viscode_correction(df):
    if 'EXAMDATE' in df.columns:
        df = df.sort_values(by=['RID','VISCODE2','EXAMDATE'])
    elif 'USERDATE' in df.columns:
        df = df.sort_values(by=['RID','VISCODE2','USERDATE'])

    pre_rid = '-1'
    viscode_list = []
    bl_viscode_date='-1'

    for index_order, row in df.iterrows():
        current_rid = row['RID']
        visdcode2 = row['VISCODE2']
        if 'EXAMDATE' in df.columns:
            current_date=row['EXAMDATE']
        elif 'USERDATE' in df.columns:
            current_date = row['USERDATE']
        else:
            return df
        if current_rid==pre_rid:
            viscode_list.append([index_order,visdcode2,current_date])
            if visdcode2=='bl':
                bl_viscode_date=current_date
        else:
            if len(viscode_list)>1:
                for i in range(1,len(viscode_list)):
                    tmp_viscode=viscode_list[i][1]
                    if not ((tmp_viscode=='bl') or (tmp_viscode=='sc') or (tmp_viscode=='scmri')):
                        if 'm' in tmp_viscode:
                            viscode_number=int(tmp_viscode.replace('m',''))
                            error_threshold=0.25#三个月
                            if (viscode_number % 6 ==0) and (viscode_number % 12 !=0):
                                error_threshold=0.5#六个月
                            elif viscode_number % 12 ==0:
                                error_threshold=1.0#12个月

                            if bl_viscode_date=='-1':
                                bl_viscode_date=viscode_list[0][2]

                            tmp_viscode_date=viscode_list[i][2]

                            bl_date = datetime.datetime.strptime(bl_viscode_date, '%Y-%m-%d %H:%M:%S')
                            current_date = datetime.datetime.strptime(tmp_viscode_date, '%Y-%m-%d %H:%M:%S')
                            delta = (current_date - bl_date).days
                            delta=delta/365.0
                            if delta>error_threshold:
                                x=1


def viscode_correction(df):#修改错误的viscode
    if 'EXAMDATE' in df.columns:
        df = df.sort_values(by=['RID','VISCODE2','EXAMDATE'])
    elif 'USERDATE' in df.columns:
        df = df.sort_values(by=['RID','VISCODE2','USERDATE'])

    pre_rid='-1'
    pre_pre_date='-1'
    pre_date='-1'
    current_date='-1'
    viscode_list=[]
    viscode_set=set()
    viscode_number_dict={}
    for index_order, row in df.iterrows():
        if 'EXAMDATE' in df.columns:
            current_date=row['EXAMDATE']
        elif 'USERDATE' in df.columns:
            current_date = row['USERDATE']
        else:
            return df
        current_rid=row['RID']
        vidcode2=row['VISCODE2']

        if current_rid==pre_rid:
            viscode_list.append([index_order,vidcode2,current_date])
            if vidcode2 in viscode_set:
                viscode_number_dict[vidcode2]=viscode_number_dict[vidcode2]+1
            else:
                viscode_set.add(vidcode2)
                viscode_number_dict[vidcode2]=1


def get_viscode_2_viscode2_dic():
    map_dic={}
    with open(viscode_2_viscode2_dic_path, 'r') as tmp_f:
        tmp_array = tmp_f.readlines()
        for i in range(1, len(tmp_array)):
            sample_data=tmp_array[i].rstrip('\n').split(',')
            rid=eval(sample_data[0])
            viscode=eval(sample_data[1])
            viscode2=sample_data[2]

            if rid not in map_dic:
                map_dic[rid]={}
            tmp_sample=map_dic[rid]
            tmp_sample[viscode]=viscode2
            map_dic[rid]=tmp_sample
        return map_dic


def keep_rows(csvname, keep_row, newcsvname) :
    f=pd.read_csv(csvname, dtype=str)
    new_f=f[keep_row]
    new_f=new_f[~new_f['RID'].isin(["999999"])]      # ~ and delete the lines including 999999

    new_f = new_f.drop_duplicates()
    new_f.to_csv(newcsvname, index = False)

def convert_medhist():
    save_file=save_path+'tmpb/clear_INITHEALTH.csv'
    user_data={}
    medhist_3_str='RID,VISCODE2,MHPSYCH,MH2NEURL_CON,MH3HEAD,MH4CARD,MH5RESP,MH6HEPAT,MH7DERM,' \
                  'MH8MUSCL,MH9ENDO,MH10GAST,MH11HEMA,MH12RENA,MH13ALLE,MH14ALCH_DRUG_SMOK,MH17MALI,MH19COG\n'
    with open(medhist_3, 'r') as tmp_f:
        tmp_array = tmp_f.readlines()
        for i in range(1, len(tmp_array)):
            sample_data=tmp_array[i].rstrip('\n').split(',')
            rid=sample_data[2]
            viscode2=sample_data[5]

            if rid not in user_data:
                user_data[rid]={}
            tmp_user=user_data[rid]
            if viscode2 not in tmp_user:
                init_str=['0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0']
                tmp_user[viscode2]=init_str
            tmp_data=tmp_user[viscode2]
            ihsymptom=eval(sample_data[9])
            if ihsymptom.isdigit():
                ihsymptom_id=int(ihsymptom)
                if ihsymptom_id!=-4:
                    if ihsymptom_id < 14:
                        tmp_data[ihsymptom_id - 1] = '1'
                    elif ihsymptom_id == 14:
                        tmp_data[13] = '1'
                    elif ihsymptom_id == 19:
                        tmp_data[15] = '1'
                    elif ihsymptom_id == 17:
                        tmp_data[14] = '1'
                    tmp_user[viscode2] = tmp_data
                    user_data[rid] = tmp_user


    for key_rid in user_data:
        tmp_patient=user_data[key_rid]
        for key_viscode2 in tmp_patient:
            tmp_recode=tmp_patient[key_viscode2]
            has_value_flage=False
            tmp_str=key_rid+','+key_viscode2
            for i in range(0,len(tmp_recode)):
                tmp_str = tmp_str+','+str(tmp_recode[i])
                if tmp_recode[i]=='1' or tmp_recode[i]==1:
                    has_value_flage=True
            if has_value_flage:
                medhist_3_str=medhist_3_str+tmp_str+"\n"
    path=os.path.dirname(save_file)
    if not os.path.exists(path):
        os.makedirs(path)
    with open(save_file, 'a') as f:
        f.write(medhist_3_str)

def convert_related():
    save_file=save_path+'tmpb/clear_FHQ.csv'
    related_str='RID,VISCODE2,MOTHALIVE,MOTHAGE,MOTHDEM,MOTHAD,MOTHSXAGE,FATHALIVE,FATHAGE,FATHDEM,FATHAD,FATHSXAGE\n'
    map_dic=get_viscode_2_viscode2_dic()
    with open(related, 'r') as tmp_f:
        tmp_array = tmp_f.readlines()
        for i in range(1, len(tmp_array)):
            sample_data=tmp_array[i].rstrip('\n').split(',')
            rid=eval(sample_data[2])
            viscode = sample_data[4]
            viscode2=viscode
            if 'v' in viscode:
                viscode=eval(viscode)
                viscode2=map_dic[rid][viscode]
            mom=sample_data[9]
            momAD=sample_data[10]
            dad=sample_data[11]
            dadAD=sample_data[12]
            related_str=related_str+sample_data[2]+','+viscode2+',-4,-4,'+mom+','+momAD+',-4,-4,-4,'+dad+','+dadAD+',-4\n'
    path=os.path.dirname(save_file)
    if not os.path.exists(path):
        os.makedirs(path)
    with open(save_file, 'a') as f:
        f.write(related_str)

#RID,VISCODE2,SIBYOB_EARLY,SIBYOB_LAST,SIBRELAT_DEMENT,SIBGENDER_DEMENT,SIBALIVE_DEMENT,SIBAGE_YOUNG,SIBAGE_OLD,SIBDEMENT,SIBAD,SIBSXAGE_YOUNG,SIBSXAGE_OLD
##SIBYOB_EARLY最早出生的兄弟姐妹年份，SIBYOB_LAST最晚出生的兄弟姐妹年份
##SIBRELAT_DEMENT患痴呆的兄弟姐妹的关系。1生理学上双亲，2生理学上母亲，3生理学上父亲，4无患病兄弟姐妹
##SIBGENDER_DEMENT患痴呆的兄弟姐妹的性别。1男性，2女性，3男女都有
##SIBALIVE_DEMENT是否还有活着的患病的兄弟姐妹。1有，0无
##SIBAGE_YOUNG,SIBAGE_OLD最年轻的兄弟姐妹年纪或死亡时的年纪，SIBYOB_LAST最老的兄弟姐妹的年纪或死亡时的年纪
##SIBDEMENT,SIBAD是否有兄弟姐妹患痴呆，以及阿兹尔海默症。
##SIBSXAGE_YOUNG,SIBSXAGE_OLD痴呆症状出现的最早年纪和最晚年纪。
def convert_re_related():
    save_file=save_path+'tmpb/clear_RECFHQ.csv'
    data_dic={}
    related_str = 'RID,VISCODE2,SIBYOB,SIBRELAT,SIBGENDER,SIBALIVE,SIBAGE,SIBDEMENT,SIBAD,SIBSXAGE,' \
                  'SIBYOB_2,SIBRELAT_2,SIBGENDER_2,SIBALIVE_2,SIBAGE_2,SIBDEMENT_2,SIBAD_2,SIBSXAGE_2,' \
                  'SIBYOB_3,SIBRELAT_3,SIBGENDER_3,SIBALIVE_3,SIBAGE_3,SIBDEMENT_3,SIBAD_3,SIBSXAGE_3,' \
                  'SIBYOB_4,SIBRELAT_4,SIBGENDER_4,SIBALIVE_4,SIBAGE_4,SIBDEMENT_4,SIBAD_4,SIBSXAGE_4,' \
                  'SIBYOB_5,SIBRELAT_5,SIBGENDER_5,SIBALIVE_5,SIBAGE_5,SIBDEMENT_5,SIBAD_5,SIBSXAGE_5,' \
                  'SIBYOB_6,SIBRELAT_6,SIBGENDER_6,SIBALIVE_6,SIBAGE_6,SIBDEMENT_6,SIBAD_6,SIBSXAGE_6,' \
                  'SIBYOB_7,SIBRELAT_7,SIBGENDER_7,SIBALIVE_7,SIBAGE_7,SIBDEMENT_7,SIBAD_7,SIBSXAGE_7,' \
                  'SIBYOB_8,SIBRELAT_8,SIBGENDER_8,SIBALIVE_8,SIBAGE_8,SIBDEMENT_8,SIBAD_8,SIBSXAGE_8,' \
                  '\n'
    #related_str='RID,VISCODE2,SIBYOB,SIBRELAT,SIBGENDER,SIBALIVE,SIBAGE,SIBDEMENT,SIBAD,SIBSXAGE\n'
    #related_str ='RID, VISCODE2, SIBYOB_EARLY, SIBYOB_LAST, SIBRELAT_DEMENT, SIBGENDER_DEMENT, SIBALIVE_DEMENT, SIBAGE_YOUNG, SIBAGE_OLD, SIBDEMENT, SIBAD, SIBSXAGE_YOUNG, SIBSXAGE_OLD'

    with open(re_related, 'r') as tmp_f:
        tmp_array = tmp_f.readlines()
        for i in range(1, len(tmp_array)):
            sample_data=tmp_array[i].rstrip('\n').split(',')
            rid=sample_data[2]
            viscode2=sample_data[5]
            tmp_key=rid+','+viscode2

            gender=sample_data[9]
            sib=sample_data[10]
            sibAD=sample_data[11]

            if not ((pd.isnull(gender) or gender=='-4' or gender=='\"-4\"' or gender=='' or gender=='\"\"') and (pd.isnull(sib) or sib=='-4' or sib=='\"-4\"' or sib=='' or sib=='\"\"') and (pd.isnull(sibAD) or sibAD=='-4' or sibAD=='\"-4\"' or sibAD=='' or sibAD=='\"\"')):
                if tmp_key not in data_dic:
                    data_dic[tmp_key] = []
                tmp_sample = data_dic[tmp_key]
                tmp_sample.append([gender, sib, sibAD])

            #related_str=related_str+rid+','+viscode2+',-4,-4,'+gender+',-4,-4,'+sib+','+sibAD+',-4\n'
    for key in data_dic:
        related_str = related_str + key
        tmp_value=data_dic[key]
        for i in range(0,len(tmp_value)):
            if i<=7:
                related_str = related_str + ',-4,-4,' + tmp_value[i][0] + ',-4,-4,' + tmp_value[i][1] + ',' + tmp_value[i][2] + ',-4'
        if len(tmp_value)<8:
            for j in range(0,8-len(tmp_value)):
                related_str = related_str + ',-4,-4,-4,-4,-4,-4,-4,-4'
        related_str=related_str+'\n'

    path=os.path.dirname(save_file)
    if not os.path.exists(path):
        os.makedirs(path)
    with open(save_file, 'a') as f:
        f.write(related_str)

def convert_recbllog():
    save_file=save_path+'tmpb/clear_RECBLLOG.csv'
    user_data={}
    recbllog_str='RID,VISCODE2,AXNAUSEA,AXVOMIT,AXDIARRH,AXCONSTP,AXABDOMN,AXSWEATN,AXDIZZY,AXENERGY,AXDROWSY,AXVISION,AXHDACHE,AXDRYMTH,AXBREATH,AXCOUGH,AXPALPIT,AXCHEST,AXURNDIS,AXURNFRQ,AXANKLE,AXMUSCLE,AXRASH,AXINSOMN,AXDPMOOD,AXCRYING,AXELMOOD,AXWANDER,AXFALL,AXOTHER\n'
    with open(recbllog, 'r') as tmp_f:
        tmp_array = tmp_f.readlines()
        for i in range(1, len(tmp_array)):
            sample_data=tmp_array[i].rstrip('\n').split(',')
            rid=sample_data[2]
            viscode2=sample_data[5]

            if rid not in user_data:
                user_data[rid]={}
            tmp_user=user_data[rid]
            if viscode2 not in tmp_user:
                init_str=['1','1','1','1','1','1','1','1','1','1','1','1','1','1','1','1','1','1','1','1','1','1','1','1','1','1','1','1']
                tmp_user[viscode2]=init_str
            tmp_data=tmp_user[viscode2]
            ihsymptom=eval(sample_data[10])
            if ihsymptom.isdigit():
                ihsymptom_id=int(ihsymptom)
                if ihsymptom_id!=-4:
                    tmp_data[ihsymptom_id - 1] = '2'
                    tmp_user[viscode2] = tmp_data
                    user_data[rid] = tmp_user


    for key_rid in user_data:
        tmp_patient=user_data[key_rid]
        for key_viscode2 in tmp_patient:
            tmp_recode=tmp_patient[key_viscode2]
            has_value_flage = False
            tmp_str=key_rid+','+key_viscode2
            for i in range(0,len(tmp_recode)):
                tmp_str = tmp_str+','+str(tmp_recode[i])
                if tmp_recode[i] == '2' or tmp_recode[i] == 2:
                    has_value_flage = True
            if has_value_flage:
                recbllog_str=recbllog_str+tmp_str+"\n"

    path=os.path.dirname(save_file)
    if not os.path.exists(path):
        os.makedirs(path)
    with open(save_file, 'a') as f:
        f.write(recbllog_str)

def clean_medhist():
    keep_rows(medhist_1go2, ['RID', 'VISCODE2', 'MHSOURCE', 'MHPSYCH', 'MH2NEURL', 'MH3HEAD', 'MH4CARD', 'MH5RESP', 'MH6HEPAT',
                             'MH7DERM', 'MH8MUSCL', 'MH9ENDO', 'MH10GAST', 'MH11HEMA', 'MH12RENA', 'MH13ALLE', 'MH14ALCH',
                             'MH14AALCH', 'MH14BALCH', 'MH14CALCH', 'MH15DRUG', 'MH15ADRUG', 'MH15BDRUG', 'MH16SMOK',
                             'MH16ASMOK', 'MH16BSMOK', 'MH16CSMOK', 'MH17MALI', 'MH18SURG'], save_path+"tmpb/clear_MEDHIST.csv")

    keep_rows(neuroexm,['RID', 'VISCODE2', 'NXVISUAL', 'NXAUDITO', 'NXTREMOR', 'NXCONSCI', 'NXNERVE', 'NXMOTOR', 'NXFINGER',
                        'NXHEEL', 'NXSENSOR', 'NXTENDON', 'NXPLANTA', 'NXGAIT', 'NXOTHER'],save_path+'tmpb/clear_NEUROEXM.csv')

    keep_rows(physical,['RID', 'VISCODE2', 'PXGENAPP', 'PXHEADEY', 'PXNECK', 'PXCHEST', 'PXHEART', 'PXABDOM', 'PXEXTREM',
                        'PXPERIPH', 'PXSKIN', 'PXMUSCUL', 'PXBACK', 'PXOTHER', 'PXABNORM'], save_path+'tmpb/clear_PHYSICAL.csv')

    keep_rows(vials, ['RID', 'VISCODE2', 'VSWEIGHT', 'VSWTUNIT', 'VSHEIGHT', 'VSHTUNIT', 'VSBPSYS', 'VSBPDIA', 'VSPULSE',
                      'VSRESP', 'VSTEMP', 'VSTMPSRC', 'VSTMPUNT'],save_path+'tmpb/clear_VITALS.csv')

    keep_rows(subject_demographics,['RID', 'VISCODE2', 'PTGENDER', 'PTDOBMM', 'PTDOBYY', 'PTHAND', 'PTMARRY', 'PTEDUCAT', 'PTWORKHS',
                                    'PTNOTRT', 'PTHOME', 'PTTLANG', 'PTPLANG', 'PTETHCAT', 'PTRACCAT'],save_path+'tmpb/clear_PTDEMOG.csv')

    keep_rows(par,['RID', 'VISCODE2', 'MOTHALIVE', 'MOTHAGE', 'MOTHDEM', 'MOTHAD', 'MOTHSXAGE', 'FATHALIVE', 'FATHAGE', 'FATHDEM',
                   'FATHAD', 'FATHSXAGE'],save_path+'tmpb/clear_FAMXHPAR.csv')

    keep_rows(sib,['RID', 'VISCODE2', 'SIBYOB', 'SIBRELAT', 'SIBGENDER', 'SIBALIVE', 'SIBAGE', 'SIBDEMENT', 'SIBAD',
                   'SIBSXAGE'],save_path+'tmpb/clear_FAMXHSIB.csv')

    keep_rows(blscheck,['RID', 'VISCODE2', 'BCNAUSEA', 'BCVOMIT', 'BCDIARRH', 'BCCONSTP', 'BCABDOMN', 'BCSWEATN', 'BCDIZZY',
                        'BCENERGY', 'BCDROWSY', 'BCVISION', 'BCHDACHE', 'BCDRYMTH', 'BCBREATH', 'BCCOUGH', 'BCPALPIT', 'BCCHEST',
                        'BCURNDIS', 'BCURNFRQ', 'BCANKLE', 'BCMUSCLE', 'BCRASH', 'BCINSOMN', 'BCDPMOOD', 'BCCRYING', 'BCELMOOD',
                        'BCWANDER', 'BCFALL', 'BCOTHER'],save_path+'tmpb/clear_BLSCHECK.csv')#与下面的相同

    keep_rows(adsxlist,['RID', 'VISCODE2', 'AXNAUSEA', 'AXVOMIT', 'AXDIARRH', 'AXCONSTP', 'AXABDOMN', 'AXSWEATN', 'AXDIZZY',
                        'AXENERGY', 'AXDROWSY', 'AXVISION', 'AXHDACHE', 'AXDRYMTH', 'AXBREATH', 'AXCOUGH', 'AXPALPIT', 'AXCHEST',
                        'AXURNDIS', 'AXURNFRQ', 'AXANKLE', 'AXMUSCLE', 'AXRASH', 'AXINSOMN', 'AXDPMOOD', 'AXCRYING', 'AXELMOOD',
                        'AXWANDER', 'AXFALL', 'AXOTHER'],save_path+'tmpb/clear_ADSXLIST.csv')

def merge_base_information():
    convert_medhist()
    convert_related()
    convert_re_related()
    convert_recbllog()

    clean_medhist()

    #病史
    df_medhist_3 = pd.read_csv(save_path+'tmpb/clear_INITHEALTH.csv', dtype=str)
    df_medhist_1go2 = pd.read_csv(save_path+"tmpb/clear_MEDHIST.csv", dtype=str)
    df_medhist=pd.concat([df_medhist_1go2,df_medhist_3],axis=0,sort=False)

    #检查
    df_neuroexm=pd.read_csv(save_path+'tmpb/clear_NEUROEXM.csv', dtype=str)
    df_physical = pd.read_csv(save_path + 'tmpb/clear_PHYSICAL.csv', dtype=str)
    df_vials = pd.read_csv(save_path + 'tmpb/clear_VITALS.csv', dtype=str)
    df_exam_2v = pd.merge(df_neuroexm, df_physical, on=['RID', 'VISCODE2'], how='outer')
    df_exam_2v = pd.merge(df_exam_2v, df_vials, on=['RID', 'VISCODE2'], how='outer')
    df_exam_2v = df_exam_2v.drop_duplicates()
    df_exam_2v.to_csv(save_path+'exam_scale.csv', index=False)

    #人口学信息
    df_subject_demographics = pd.read_csv(save_path + 'tmpb/clear_PTDEMOG.csv', dtype=str)
    df_par_3 = pd.read_csv(save_path + 'tmpb/clear_FAMXHPAR.csv', dtype=str)
    df_par_1go2 = pd.read_csv(save_path + 'tmpb/clear_FHQ.csv', dtype=str)
    df_par=pd.concat([df_par_1go2,df_par_3],axis=0,sort=False)

    df_sib_3 = pd.read_csv(save_path + 'tmpb/clear_FAMXHSIB.csv', dtype=str)
    df_sib_1go2 = pd.read_csv(save_path + 'tmpb/clear_RECFHQ.csv', dtype=str)
    df_sib = pd.concat([df_sib_1go2, df_sib_3], axis=0, sort=False)

    #症状
    df_blscheck = pd.read_csv(save_path + 'tmpb/clear_BLSCHECK.csv', dtype=str)
    df_blscheck.rename(columns={'BCNAUSEA':'AXNAUSEA', 'BCVOMIT':'AXVOMIT', 'BCDIARRH':'AXDIARRH', 'BCCONSTP':'AXCONSTP', 'BCABDOMN':'AXABDOMN', 'BCSWEATN':'AXSWEATN', 'BCDIZZY':'AXDIZZY',
                        'BCENERGY':'AXENERGY', 'BCDROWSY':'AXDROWSY', 'BCVISION':'AXVISION', 'BCHDACHE':'AXHDACHE', 'BCDRYMTH':'AXDRYMTH', 'BCBREATH':'AXBREATH', 'BCCOUGH':'AXCOUGH', 'BCPALPIT':'AXPALPIT', 'BCCHEST':'AXCHEST',
                        'BCURNDIS':'AXURNDIS', 'BCURNFRQ':'AXURNFRQ', 'BCANKLE':'AXANKLE', 'BCMUSCLE':'AXMUSCLE', 'BCRASH':'AXRASH', 'BCINSOMN':'AXINSOMN', 'BCDPMOOD':'AXDPMOOD', 'BCCRYING':'AXCRYING', 'BCELMOOD':'AXELMOOD',
                        'BCWANDER':'AXWANDER', 'BCFALL':'AXFALL', 'BCOTHER':'AXOTHER'} ,inplace=True)
    df_adsxlist = pd.read_csv(save_path + 'tmpb/clear_ADSXLIST.csv', dtype=str)
    df_recbllog = pd.read_csv(save_path + 'tmpb/clear_RECBLLOG.csv', dtype=str)

    df_blscheck = df_blscheck.append(df_adsxlist)
    df_blscheck = df_blscheck.append(df_recbllog)
    df_blscheck = df_blscheck.drop_duplicates(subset=['RID', 'VISCODE2'])
    df_symptom = df_blscheck


    df_base = pd.merge(df_subject_demographics, df_medhist, on=['RID', 'VISCODE2'], how='outer')
    df_family = pd.merge(df_par, df_sib, on=['RID', 'VISCODE2'], how='outer')


    df_base=pd.merge(df_base, df_family, on=['RID', 'VISCODE2'], how='outer')
    df_base = pd.merge(df_base, df_symptom, on=['RID', 'VISCODE2'], how='outer')
    #df_base = pd.merge(df_base, df_exam, on=['RID', 'VISCODE2'], how='outer')

    df_base = df_base.drop_duplicates()
    df_base.to_csv(save_path+'base_scale.csv', index=False)

def combin_npi():
    f_npi=pd.read_csv(npi, dtype=str)
    new_npi=f_npi[['RID', 'VISCODE2', 'NPIA', 'NPIA10B', 'NPIB', 'NPIB8B', 'NPIC', 'NPIC9B', 'NPID', 'NPID9B',
                  'NPIE', 'NPIE8B', 'NPIF', 'NPIF8B', 'NPIG', 'NPIG9B', 'NPIH', 'NPIH8B', 'NPII',
                  'NPII8B', 'NPIJ', 'NPIJ8B', 'NPIK', 'NPIK9B', 'NPIL', 'NPIL9B']]
    f_npiq=pd.read_csv(npiq, dtype=str)
    new_npiq=f_npiq[['RID', 'VISCODE2', 'NPIA', 'NPIASEV', 'NPIB', 'NPIBSEV', 'NPIC', 'NPICSEV', 'NPID', 'NPIDSEV',
                  'NPIE', 'NPIESEV', 'NPIF', 'NPIFSEV', 'NPIG', 'NPIGSEV', 'NPIH', 'NPIHSEV', 'NPII',
                  'NPIISEV', 'NPIJ', 'NPIJSEV', 'NPIK', 'NPIKSEV', 'NPIL', 'NPILSEV']]

    f_npi.rename(columns={'NPIA10B':'NPIASEV', 'NPIB8B':'NPIBSEV', 'NPIC9B':'NPICSEV', 'NPID9B':'NPIDSEV',
                  'NPIE8B':'NPIESEV', 'NPIF8B':'NPIFSEV', 'NPIG9B':'NPIGSEV', 'NPIH8B':'NPIHSEV', 'NPII8B':
                  'NPIISEV', 'NPIJ8B':'NPIJSEV', 'NPIK9B':'NPIKSEV', 'NPIL9B':'NPILSEV'},inplace=True)
    npi_com =pd.concat([new_npi ,new_npiq], sort=False)
    #npi_com = npi_com.set_index(['RID', 'VISCODE2'], drop=False)
    return npi_com

def get_ecogsp():
    ecogsp_f=pd.read_csv(ecogsp, dtype=str)
    new_ecogsp=ecogsp_f[['RID', 'VISCODE2', 'MEMORY1', 'MEMORY2', 'MEMORY3', 'MEMORY4', 'MEMORY5', 'MEMORY6',
                    'MEMORY7', 'MEMORY8', 'LANG1', 'LANG2', 'LANG3', 'LANG4', 'LANG5', 'LANG6', 'LANG7',
                    'LANG8', 'LANG9', 'VISSPAT1', 'VISSPAT2', 'VISSPAT3', 'VISSPAT4', 'VISSPAT5', 'VISSPAT6', 'VISSPAT7',
                    'VISSPAT8', 'PLAN1', 'PLAN2', 'PLAN3', 'PLAN4', 'PLAN5', 'ORGAN1', 'ORGAN2', 'ORGAN3',
                    'ORGAN4', 'ORGAN5', 'ORGAN6', 'DIVATT1', 'DIVATT2', 'DIVATT3', 'DIVATT4',
                    'SOURCE', 'EcogSPDivatt', 'EcogSPLang', 'EcogSPMem', 'EcogSPOrgan', 'EcogSPPlan', 'EcogSPVisspat',
                    'EcogSPTotal']]
    new_ecogsp.rename(columns={'MEMORY1':'MEMORY1_sp', 'MEMORY2':'MEMORY2_sp', 'MEMORY3':'MEMORY3_sp', 'MEMORY4':'MEMORY4_sp', 'MEMORY5':'MEMORY5_sp', 'MEMORY6':'MEMORY6_sp',
                    'MEMORY7':'MEMORY7_sp', 'MEMORY8':'MEMORY8_sp', 'LANG1':'LANG1_sp', 'LANG2':'LANG2_sp', 'LANG3':'LANG3_sp', 'LANG4':'LANG4_sp', 'LANG5':'LANG5_sp', 'LANG6':'LANG6_sp', 'LANG7':'LANG7_sp',
                    'LANG8':'LANG8_sp', 'LANG9':'LANG9_sp', 'VISSPAT1':'VISSPAT1_sp', 'VISSPAT2':'VISSPAT2_sp', 'VISSPAT3':'VISSPAT3_sp', 'VISSPAT4':'VISSPAT4_sp', 'VISSPAT5':'VISSPAT5_sp', 'VISSPAT6':'VISSPAT6_sp', 'VISSPAT7':'VISSPAT7_sp',
                    'VISSPAT8':'VISSPAT8_sp', 'PLAN1':'PLAN1_sp', 'PLAN2':'PLAN2_sp', 'PLAN3':'PLAN3_sp', 'PLAN4':'PLAN4_sp', 'PLAN5':'PLAN5_sp', 'ORGAN1':'ORGAN1_sp', 'ORGAN2':'ORGAN2_sp', 'ORGAN3':'ORGAN3_sp',
                    'ORGAN4':'ORGAN4_sp', 'ORGAN5':'ORGAN5_sp', 'ORGAN6':'ORGAN6_sp', 'DIVATT1':'DIVATT1_sp', 'DIVATT2':'DIVATT2_sp', 'DIVATT3':'DIVATT3_sp', 'DIVATT4':'DIVATT4_sp', 'SOURCE':'SOURCE_sp'},inplace=True)
    return new_ecogsp


# 阿尔兹海默症评价量表-认知分表
# ADAS-Cog分为12个条目，语词回忆、命名、执行指令、结构性练习、意向性练习、
# 定向力、词语辨认、回忆测验指令、口头语言表达能力、找词能力、语言理解能力和注意力。
# 从记忆、语言、操作能力和注意力4个方面评估认知能力。评分范围为0~75分，分数越高认知受损越严重
def get_adas():
    adni_1 =pd.DataFrame(pd.read_csv(Neur_path +'ADAS_ADNI1.csv', dtype=str))

    #####
    # hand revise  ADNI3  6712 bl
    #                     6315 m12
    #      revise  VISCODE2  RID=5127,5083,4198,5200,4552,4028,5126,4271,6143,6160,6708
    adni_other =pd.DataFrame(pd.read_csv(Neur_path +'ADAS_ADNIGO23_NEW.csv', dtype=str))

    # adni_other.drop(['Phase','ID','SITEID','VISCODE','USERDATE','USERDATE2','','','','','','','','','','','',''],axis=1,inplace=True)
    adni_other =adni_other[['Phase', 'RID', 'VISCODE2', 'Q1UNABLE', 'Q1TR1', 'Q1TR2', 'Q1TR3', 'Q2UNABLE', 'Q2TASK', 'Q3UNABLE',
                           'Q3TASK1', 'Q3TASK2', 'Q3TASK3', 'Q3TASK4', 'Q4UNABLE', 'Q4TASK', 'Q5UNABLE', 'Q5TASK',
                           'Q5NAME1', 'Q5NAME2', 'Q5NAME3', 'Q5NAME4', 'Q5NAME5', 'Q5NAME6', 'Q5NAME7', 'Q5NAME8', 'Q5NAME9',
                           'Q5NAME10', 'Q5NAME11', 'Q5NAME12', 'Q5FINGER', 'Q6UNABLE', 'Q6TASK', 'Q7UNABLE',
                           'Q7TASK', 'Q8UNABLE', 'Q8WORD1', 'Q8WORD1R', 'Q8WORD2', 'Q8WORD2R', 'Q8WORD3', 'Q8WORD3R', 'Q8WORD4',
                           'Q8WORD4R', 'Q8WORD5', 'Q8WORD5R', 'Q8WORD6', 'Q8WORD6R', 'Q8WORD7', 'Q8WORD7R', 'Q8WORD8', 'Q8WORD8R',
                           'Q8WORD9', 'Q8WORD9R', 'Q8WORD10', 'Q8WORD10R', 'Q8WORD11', 'Q8WORD11R', 'Q8WORD12', 'Q8WORD12R',
                           'Q8WORD13', 'Q8WORD13R', 'Q8WORD14', 'Q8WORD14R', 'Q8WORD15', 'Q8WORD15R', 'Q8WORD16', 'Q8WORD16R',
                           'Q8WORD17', 'Q8WORD17R', 'Q8WORD18', 'Q8WORD18R', 'Q8WORD19', 'Q8WORD19R', 'Q8WORD20', 'Q8WORD20R',
                           'Q8WORD21', 'Q8WORD21R', 'Q8WORD22', 'Q8WORD22R', 'Q8WORD23', 'Q8WORD23R', 'Q8WORD24', 'Q8WORD24R',
                           'Q9TASK', 'Q10TASK', 'Q11TASK', 'Q12TASK', 'Q13UNABLE', 'Q13TASKA', 'Q13TASKB', 'Q13TASKC']]

    col_name = adni_other.columns.tolist()
    q1_1_list =['Q1_1_0' ,'Q1_1_1' ,'Q1_1_2' ,'Q1_1_3' ,'Q1_1_4' ,'Q1_1_5' ,'Q1_1_6' ,'Q1_1_7' ,'Q1_1_8' ,'Q1_1_9']
    [col_name.insert(col_name.index('Q1TR2'), d) for d in q1_1_list]

    q1_2_list =['Q1_2_0' ,'Q1_2_1' ,'Q1_2_2' ,'Q1_2_3' ,'Q1_2_4' ,'Q1_2_5' ,'Q1_2_6' ,'Q1_2_7' ,'Q1_2_8' ,'Q1_2_9']
    [col_name.insert(col_name.index('Q1TR3'), d) for d in q1_2_list]

    q1_3_list =['Q1_3_0' ,'Q1_3_1' ,'Q1_3_2' ,'Q1_3_3' ,'Q1_3_4' ,'Q1_3_5' ,'Q1_3_6' ,'Q1_3_7' ,'Q1_3_8' ,'Q1_3_9']
    [col_name.insert(col_name.index('Q2UNABLE'), d) for d in q1_3_list]
    # adni_other=adni_other.reindex(columns=col_name)

    q2_list =['Q2_1' ,'Q2_2' ,'Q2_3' ,'Q2_4' ,'Q2_5']
    [col_name.insert(col_name.index('Q3UNABLE'), d) for d in q2_list]
    # adni_other=adni_other.reindex(columns=col_name)

    q3_list =['Q3TASK']
    [col_name.insert(col_name.index('Q3TASK1'), d) for d in q3_list]

    q4_list =['Q4_0' ,'Q4_1' ,'Q4_2' ,'Q4_3' ,'Q4_4' ,'Q4_5' ,'Q4_6' ,'Q4_7' ,'Q4_8' ,'Q4_9']
    [col_name.insert(col_name.index('Q5UNABLE'), d) for d in q4_list]

    q5_list =['Q5NAME13' ,'Q5NAME14' ,'Q5NAME15' ,'Q5NAME16' ,'Q5NAME17']
    [col_name.insert(col_name.index('Q6UNABLE'), d) for d in q5_list]

    q6_list =['Q6_1' ,'Q6_2' ,'Q6_3' ,'Q6_4' ,'Q6_5']
    [col_name.insert(col_name.index('Q7UNABLE'), d) for d in q6_list]

    q7_list =['Q7_1' ,'Q7_2' ,'Q7_3' ,'Q7_4' ,'Q7_5' ,'Q7_6' ,'Q7_7' ,'Q7_8']
    [col_name.insert(col_name.index('Q8UNABLE'), d) for d in q7_list]

    adni_other =adni_other.reindex(columns=col_name)

    adni_1 =adni_1[['RID', 'VISCODE', 'COT1LIST', 'COT2LIST', 'COT3LIST', 'COP1COMP', 'COCOMND', 'COP2COMP', 'COCONSTR',
                   'COP3COMP', 'COT4LIST', 'COP4COMP', 'CONAME', 'COP5COMP', 'COIDEA', 'COP6COMP', 'COORIEN', 'COP7COMP',
                   'CO8NURSE', 'CO8MAGAZ', 'CO8WIZRD', 'CO8WIZRR', 'CO8VAN', 'CO8VANR', 'CO8LEPRD', 'CO8LEPRR', 'CO8SALE',
                   'CO8SALER', 'CO8SEA', 'CO8SEAR', 'CO8TRAIN', 'CO8TRAIR', 'CO8COIN', 'CO8COINR', 'CO8SHIP', 'CO8SHIPR',
                   'CO8INST', 'CO8INSTR', 'CO8MAP', 'CO8MAPR', 'CO8AXE', 'CO8AXER', 'CO8BOARD', 'CO8BOARR', 'CO8CARRT',
                   'CO8CARRR', 'CO8MILK', 'CO8MILKR', 'CO8VOL', 'CO8VOLR', 'CO8FORST', 'CO8FORSR', 'CO8ANCHR', 'CO8ANCRR',
                   'CO8GEM', 'CO8GEMR', 'CO8CAT', 'CO8CATR', 'CO8FUND', 'CO8FUNDR', 'CO8EDGE', 'CO8EDGER', 'CO8CAKE',
                   'CO8CAKER', 'COP8COMP', 'COINSTRC', 'COCOMPRE', 'COWRDFND', 'COLANG', 'CONMCXLA', 'CONMCXLB', 'CONMCXLC', 'COP14CMP']]
    adni_1.rename(columns={'VISCODE' :'VISCODE2', 'COT1LIST' :'Q1TR1', 'COT2LIST' :'Q1TR2', 'COT3LIST' :'Q1TR3', 'COP1COMP' :'Q1UNABLE',
                           'COCOMND' :'Q2TASK', 'COP2COMP' :'Q2UNABLE', 'COCONSTR': 'Q3TASK', 'COP3COMP' :'Q3UNABLE',
                           'COT4LIST' :'Q4TASK', 'COP4COMP' :'Q4UNABLE', 'CONAME' :'Q5TASK', 'COP5COMP' :'Q5UNABLE',
                           'COIDEA' :'Q6TASK', 'COP6COMP' :'Q6UNABLE',
                           'COORIEN' :'Q7TASK', 'COP7COMP' :'Q7UNABLE',
                           'CO8NURSE' :'Q8WORD1',
                           'CO8MAGAZ' :'Q8WORD2',
                           'CO8WIZRD' :'Q8WORD3',
                           'CO8WIZRR' :'Q8WORD3R',
                           'CO8VAN' :'Q8WORD4',
                           'CO8VANR' :'Q8WORD4R',
                           'CO8LEPRD' :'Q8WORD5',
                           'CO8LEPRR' :'Q8WORD5R',
                           'CO8SALE' :'Q8WORD6',
                           'CO8SALER' :'Q8WORD6R',
                           'CO8SEA' :'Q8WORD7',
                           'CO8SEAR' :'Q8WORD7R',
                           'CO8TRAIN' :'Q8WORD8',
                           'CO8TRAIR' :'Q8WORD8R',
                           'CO8COIN' :'Q8WORD9',
                           'CO8COINR' :'Q8WORD9R',
                           'CO8SHIP' :'Q8WORD10',
                           'CO8SHIPR' :'Q8WORD10R',
                           'CO8INST' :'Q8WORD11',
                           'CO8INSTR' :'Q8WORD11R',
                           'CO8MAP' :'Q8WORD12',
                           'CO8MAPR' :'Q8WORD12R',
                           'CO8AXE' :'Q8WORD13',
                           'CO8AXER' :'Q8WORD13R',
                           'CO8BOARD' :'Q8WORD14',
                           'CO8BOARR' :'Q8WORD14R',
                           'CO8CARRT' :'Q8WORD15',
                           'CO8CARRR' :'Q8WORD15R',
                           'CO8MILK' :'Q8WORD16',
                           'CO8MILKR' :'Q8WORD16R',
                           'CO8VOL' :'Q8WORD17',
                           'CO8VOLR' :'Q8WORD17R',
                           'CO8FORST' :'Q8WORD18',
                           'CO8FORSR' :'Q8WORD18R',
                           'CO8ANCHR' :'Q8WORD19',
                           'CO8ANCRR' :'Q8WORD19R',
                           'CO8GEM' :'Q8WORD20',
                           'CO8GEMR' :'Q8WORD20R',
                           'CO8CAT' :'Q8WORD21',
                           'CO8CATR' :'Q8WORD21R',
                           'CO8FUND' :'Q8WORD22',
                           'CO8FUNDR' :'Q8WORD22R',
                           'CO8EDGE' :'Q8WORD23',
                           'CO8EDGER' :'Q8WORD23R',
                           'CO8CAKE' :'Q8WORD24',
                           'CO8CAKER' :'Q8WORD24R',
                           'COP8COMP' :'Q8UNABLE',
                           'COINSTRC': 'Q9TASK',
                           'COCOMPRE': 'Q10TASK',
                           'COWRDFND': 'Q11TASK',
                           'COLANG': 'Q12TASK',
                           'CONMCXLA' :'Q13TASKA', 'CONMCXLB' :'Q13TASKB', 'CONMCXLC' :'Q13TASKC', 'COP14CMP' :'Q13UNABLE'} ,inplace=True)
    adni =pd.concat([adni_other ,adni_1], sort=False)
    adni = adni.set_index(['RID', 'VISCODE2'], drop=False)

    for index_order, row in adni.iterrows():
        phase=row['Phase']
        key1=row['RID']
        key2=row['VISCODE2']
        #Q1
        for j in range(1,4):
            q1_str='Q1TR'+str(j)
            q1=row[q1_str]
            if pd.isnull(q1) or q1=='-4':
                for i in range(0,10):
                    c_str = 'Q1_'+str(j)+'_' + str(i)
                    adni.loc[(key1,key2), c_str]=-4
            else:
                #print(row['RID'])
                #print(row['VISCODE2'])
                #print(q1)
                #print('******'+q1)
                r1_list=re.split(':|\|',q1)
                index_set=set()
                for sub_index in range(0,len(r1_list)):
                    word_id=int(r1_list[sub_index])
                    if pd.isnull(phase):
                        word_id=word_id-1
                    index_set.add(word_id)

                for i in range(0,10):
                    c_str = 'Q1_'+str(j)+'_' + str(i)
                    if i in index_set:
                        adni.loc[(key1,key2),c_str]=1
                    else:
                        adni.loc[(key1,key2),c_str]=0
                #kk=np.array(adni.loc[0,:]).tolist()
                #print(kk)

        #Q2,Q4,Q6,Q7
        datalist=['Q2TASK','Q4TASK','Q6TASK','Q7TASK']
        datalen=[5,10,5,8]
        for i in range(0,len(datalist)):
            c_str=datalist[i]
            tmp_len=datalen[i]

            r_str=row[c_str]
            if pd.isnull(r_str) or r_str=='-4':
                for j in range(1,tmp_len+1):
                    tmpj=j
                    if i==1:
                        tmpj=tmpj-1
                    tmp_str=c_str[0:2]+'_'+str(tmpj)
                    adni.loc[(key1,key2), tmp_str] = -4
            else:
                r_list=re.split(':|\|',r_str)
                index_set=set()
                for sub_index in range(0,len(r_list)):
                    word_id=int(r_list[sub_index])
                    if pd.isnull(phase) and i==1:
                        word_id=word_id-1
                    index_set.add(word_id)

                for j in range(1,tmp_len+1):
                    tmpj=j
                    if i==1:
                        tmpj=tmpj-1
                    tmp_str=c_str[0:2]+'_'+str(tmpj)

                    if tmpj in index_set:
                        adni.loc[(key1,key2), tmp_str] =1
                    else:
                        adni.loc[(key1,key2), tmp_str] =0

        #Q3 convert ADNI1 to ADNIGO23
        if pd.isnull(phase): #ADNI1
            q3=row['Q3TASK']
            if pd.isnull(q3) or q3=='-4':
                for j in range(1,5):
                    tmp_str='Q3TASK'+str(j)
                    adni.loc[(key1,key2), tmp_str] = -4
            else:
                r3_list = re.split(':|\|', q3)
                index_set = set()
                for sub_index in range(0, len(r3_list)):
                    word_id = int(r3_list[sub_index])
                    index_set.add(word_id)

                for i in range(2, 6):
                    tmp_str='Q3TASK'+str(i-1)
                    if 1 in index_set:
                        adni.loc[(key1,key2), tmp_str] = 2
                    elif i in index_set:
                        adni.loc[(key1,key2), tmp_str] = 1
                    else:
                        adni.loc[(key1,key2), tmp_str] = -4

        #Q5
        if phase=='ADNI3':
            q5 = row['Q5FINGER']
            if pd.isnull(q5) or q5 == '-4':
                for i in range(13, 18):
                    c_str = 'Q5NAME' + str(i)
                    adni.loc[(key1,key2), c_str] = -4
            else:
                r5_list = re.split(':|\|', q5)
                index_set = set()
                for sub_index in range(0, len(r5_list)):
                    word_id = int(r5_list[sub_index])
                    index_set.add(word_id)

                for i in range(13, 18):
                    c_str = 'Q5NAME' + str(i)
                    if (i-12) in index_set:
                        adni.loc[(key1,key2), c_str] = 1
                    else:
                        adni.loc[(key1,key2), c_str] = 0
        else:
            q5 = row['Q5TASK']
            if pd.isnull(q5) or q5 == '-4':
                for i in range(1, 18):
                    c_str = 'Q5NAME'+str(i)
                    adni.loc[(key1,key2), c_str] = -4
            else:
                r5_list = re.split(':|\|', q5)
                index_set = set()
                for sub_index in range(0, len(r5_list)):
                    word_id = int(r5_list[sub_index])
                    index_set.add(word_id)

                for i in range(1, 18):
                    c_str = 'Q5NAME'+str(i)
                    if i in index_set:
                        adni.loc[(key1,key2), c_str] = 1
                    else:
                        adni.loc[(key1,key2), c_str] = 0
    adni = adni.reset_index(drop=True)
    return adni[['RID', 'VISCODE2', 'Q1UNABLE', 'Q1_1_0', 'Q1_1_1', 'Q1_1_2', 'Q1_1_3', 'Q1_1_4', 'Q1_1_5', 'Q1_1_6',
                'Q1_1_7', 'Q1_1_8', 'Q1_1_9', 'Q1_2_0', 'Q1_2_1', 'Q1_2_2', 'Q1_2_3', 'Q1_2_4', 'Q1_2_5', 'Q1_2_6', 'Q1_2_7',
                'Q1_2_8', 'Q1_2_9', 'Q1_3_0', 'Q1_3_1', 'Q1_3_2', 'Q1_3_3', 'Q1_3_4', 'Q1_3_5', 'Q1_3_6', 'Q1_3_7', 'Q1_3_8',
                'Q1_3_9', 'Q2UNABLE', 'Q2_1', 'Q2_2', 'Q2_3', 'Q2_4', 'Q2_5', 'Q3UNABLE', 'Q3TASK1', 'Q3TASK2', 'Q3TASK3',
                'Q3TASK4', 'Q4UNABLE', 'Q4_0', 'Q4_1', 'Q4_2', 'Q4_3', 'Q4_4', 'Q4_5', 'Q4_6', 'Q4_7', 'Q4_8', 'Q4_9',
                'Q5UNABLE', 'Q5NAME1', 'Q5NAME2', 'Q5NAME3', 'Q5NAME4', 'Q5NAME5', 'Q5NAME6', 'Q5NAME7', 'Q5NAME8', 'Q5NAME9',
                'Q5NAME10', 'Q5NAME11', 'Q5NAME12', 'Q5NAME13', 'Q5NAME14', 'Q5NAME15', 'Q5NAME16', 'Q5NAME17', 'Q6UNABLE',
                'Q6_1', 'Q6_2', 'Q6_3', 'Q6_4', 'Q6_5', 'Q7UNABLE','Q7_1', 'Q7_2', 'Q7_3', 'Q7_4', 'Q7_5', 'Q7_6', 'Q7_7',
                'Q7_8', 'Q8UNABLE', 'Q8WORD1', 'Q8WORD1R', 'Q8WORD2', 'Q8WORD2R', 'Q8WORD3', 'Q8WORD3R', 'Q8WORD4', 'Q8WORD4R',
                'Q8WORD5', 'Q8WORD5R', 'Q8WORD6', 'Q8WORD6R', 'Q8WORD7', 'Q8WORD7R', 'Q8WORD8', 'Q8WORD8R', 'Q8WORD9', 'Q8WORD9R',
                'Q8WORD10', 'Q8WORD10R', 'Q8WORD11', 'Q8WORD11R', 'Q8WORD12', 'Q8WORD12R', 'Q8WORD13', 'Q8WORD13R', 'Q8WORD14',
                'Q8WORD14R', 'Q8WORD15', 'Q8WORD15R', 'Q8WORD16', 'Q8WORD16R', 'Q8WORD17', 'Q8WORD17R', 'Q8WORD18', 'Q8WORD18R',
                'Q8WORD19', 'Q8WORD19R', 'Q8WORD20', 'Q8WORD20R', 'Q8WORD21', 'Q8WORD21R', 'Q8WORD22', 'Q8WORD22R', 'Q8WORD23',
                'Q8WORD23R', 'Q8WORD24', 'Q8WORD24R', 'Q9TASK', 'Q10TASK', 'Q11TASK', 'Q12TASK', 'Q13UNABLE', 'Q13TASKA',
                'Q13TASKB', 'Q13TASKC']]

def clean_cog():
    path=os.path.dirname(save_path + 'tmpcog/clear_MMSE.csv')
    if not os.path.exists(path):
        os.makedirs(path)
    keep_rows(mmse,['RID', 'VISCODE2', 'MMDATE', 'MMYEAR', 'MMMONTH', 'MMDAY', 'MMSEASON', 'MMHOSPIT', 'MMFLOOR', 'MMCITY',
                    'MMAREA', 'MMSTATE', 'MMRECALL', 'MMBALL', 'MMFLAG', 'MMTREE', 'MMTRIALS', 'MMD', 'MML',
                    'MMR', 'MMO', 'MMW', 'MMBALLDL', 'MMFLAGDL', 'MMTREEDL', 'MMWATCH', 'MMPENCIL', 'MMREPEAT', 'MMHAND', 'MMFOLD',
                    'MMONFLR', 'MMREAD', 'MMWRITE', 'MMDRAW', 'MMSCORE', 'DONE', 'WORD1', 'WORD1DL', 'WORD2', 'WORD2DL',
                    'WORD3', 'WORD3DL', 'WORDLIST', 'WORLDSCORE'],save_path + 'tmpcog/clear_MMSE.csv')

    keep_rows(moca,['RID', 'VISCODE2','TRAILS', 'CUBE','CLOCKCON', 'CLOCKNO','CLOCKHAN', 'LION','RHINO', 'CAMEL','IMMT1W1', 'IMMT1W2',
                    'IMMT1W3', 'IMMT1W4','IMMT1W5', 'IMMT2W1','IMMT2W2', 'IMMT2W3','IMMT2W4', 'IMMT2W5','DIGFOR', 'DIGBACK','LETTERS', 'SERIAL1',
                    'SERIAL2', 'SERIAL3','SERIAL4', 'SERIAL5','REPEAT1', 'REPEAT2','FFLUENCY', 'ABSTRAN','ABSMEAS', 'DELW1','DELW2', 'DELW3',
                    'DELW4','DELW5','DATE','MONTH','YEAR','DAY','PLACE','CITY','MOCA'
                    ],save_path + 'tmpcog/clear_MOCA.csv')
    #移动到认知
    keep_rows(cdr,['RID', 'VISCODE2', 'CDVERSION', 'CDMEMORY', 'CDORIENT', 'CDJUDGE', 'CDCOMMUN', 'CDHOME', 'CDCARE',
                   'CDGLOBAL'],save_path+'tmpcog/clear_CDR.csv')
    keep_rows(cci, ['RID', 'VISCODE2', 'CCI1', 'CCI2', 'CCI3', 'CCI4', 'CCI5', 'CCI6', 'CCI7', 'CCI8', 'CCI9', 'CCI10',
                    'CCI11', 'CCI12', 'CCI13', 'CCI14', 'CCI15', 'CCI16', 'CCI17', 'CCI18', 'CCI19', 'CCI20', 'CCI12TOT'],
              save_path + 'tmpcog/clear_CCI.csv')

#认知
def merge_cog_information():
    df_adas=get_adas()
    clean_cog()

    df_mmse=pd.read_csv(save_path + 'tmpcog/clear_MMSE.csv', dtype=str)
    df_moca=pd.read_csv(save_path + 'tmpcog/clear_MOCA.csv', dtype=str)
    df_cdr = pd.read_csv(save_path + 'tmpcog/clear_CDR.csv', dtype=str)
    df_cci = pd.read_csv(save_path + 'tmpcog/clear_CCI.csv', dtype=str)

    df_cog=pd.merge(df_adas, df_mmse, on=['RID', 'VISCODE2'], how='outer')
    df_cog = pd.merge(df_cog, df_moca, on=['RID', 'VISCODE2'], how='outer')
    df_cog = pd.merge(df_cog, df_cdr, on=['RID', 'VISCODE2'], how='outer')
    df_cog = pd.merge(df_cog, df_cci, on=['RID', 'VISCODE2'], how='outer')
    df_cog = df_cog.drop_duplicates()
    df_cog.to_csv(save_path + 'cog_scale.csv', index=False)

#'SERIAL2', 'SERIAL3','SERIAL4', 'SERIAL5','REPEAT1', 'REPEAT2','FFLUENCY', 'ABSTRAN','ABSMEAS', 'DELW1','DELW2', 'DELW3',
#时钟绘画测试、逻辑记忆测试-I（即时回忆）、逻辑记忆测试-II（延时回忆）、雷伊听觉测试、连线测试、动物类别测试、波士顿命名测试、美国成人阅读测试。（
def clean_ntest():
    path=os.path.dirname(save_path + 'tmpN/clear_NEUROBAT.csv')
    if not os.path.exists(path):
        os.makedirs(path)
    keep_rows(neurobat,['RID', 'VISCODE2','CLOCKCIRC', 'CLOCKSYM','CLOCKNUM', 'CLOCKHAND','CLOCKTIME', 'CLOCKSCOR','COPYCIRC', 'COPYSYM','COPYNUM', 'COPYHAND',
                    'COPYTIME', 'COPYSCOR','LMSTORY', 'LIMMTOTAL','LIMMEND', 'AVTOT1','AVERR1', 'AVTOT2','AVERR2', 'AVTOT3','AVERR3', 'AVTOT4',
                    'AVERR4','AVTOT5','AVERR5','AVTOT6','AVERR6','AVTOTB','AVERRB','AVENDED','DSPANFOR','DSPANFLTH','DSPANBAC','DSPANBLTH','CATANIMSC','CATANPERS','CATANINTR',
                    'CATVEGESC','CATVGPERS','CATVGINTR','TRAASCOR','TRAAERRCOM','TRAAERROM','TRABSCOR','TRABERRCOM','TRABERROM','DIGITSCOR','LDELBEGIN','LDELTOTAL','LDELCUE','BNTND','BNTSPONT','BNTSTIM',
                    'BNTCSTIM','BNTPHON','BNTCPHON','BNTTOTAL','AVDELBEGAN','AVDEL30MIN','AVDELERR1','AVDELTOT','AVDELERR2','ANARTND','ANARTERR','MINTSEMCUE','MINTTOTAL','MINTUNCUED','RAVLT_forgetting',
                    'RAVLT_immediate','RAVLT_learning','RAVLT_perc_forgetting'
                    ],save_path + 'tmpN/clear_NEUROBAT.csv')

#用于鉴别SMC
def clean_cci():
    path=os.path.dirname(save_path+'tmpSMC/clear_CCI.csv')
    if not os.path.exists(path):
        os.makedirs(path)
    keep_rows(cci,['RID', 'VISCODE2', 'CCI1', 'CCI2', 'CCI3', 'CCI4', 'CCI5', 'CCI6', 'CCI7', 'CCI8', 'CCI9', 'CCI10', 'CCI11', 'CCI12', 'CCI13',
                   'CCI14', 'CCI15', 'CCI16', 'CCI17', 'CCI18', 'CCI19', 'CCI20', 'CCI12TOT'],save_path+'tmpSMC/clear_CCI.csv')

def clean_nfb():#功能、行为、精神
    path=os.path.dirname(save_path+'tmpfb/clear_CDR.csv')
    if not os.path.exists(path):
        os.makedirs(path)

    keep_rows(gds, ['RID', 'VISCODE2', 'GDUNABL', 'GDSATIS', 'GDDROP', 'GDEMPTY', 'GDBORED', 'GDSPIRIT', 'GDAFRAID',
                    'GDHAPPY', 'GDHELP', 'GDHOME', 'GDMEMORY', 'GDALIVE', 'GDWORTH', 'GDENERGY', 'GDHOPE', 'GDBETTER',
                    'GDTOTAL'], save_path + 'tmpfb/clear_GDSCALE.csv')

    keep_rows(faq, ['RID', 'VISCODE2', 'FAQFINAN', 'FAQFORM', 'FAQSHOP', 'FAQGAME', 'FAQBEVG', 'FAQMEAL', 'FAQEVENT',
                    'FAQTV', 'FAQREM', 'FAQTRAVL', 'FAQTOTAL'], save_path + 'tmpfb/clear_FAQ.csv')

    keep_rows(ecogpt, ['RID', 'VISCODE2', 'CONCERN', 'MEMORY1', 'MEMORY2', 'MEMORY3', 'MEMORY4', 'MEMORY5', 'MEMORY6',
                    'MEMORY7', 'MEMORY8', 'LANG1', 'LANG2', 'LANG3', 'LANG4', 'LANG5', 'LANG6', 'LANG7',
                    'LANG8', 'LANG9', 'VISSPAT1', 'VISSPAT2', 'VISSPAT3', 'VISSPAT4', 'VISSPAT5', 'VISSPAT6', 'VISSPAT7',
                    'VISSPAT8', 'PLAN1', 'PLAN2', 'PLAN3', 'PLAN4', 'PLAN5', 'ORGAN1', 'ORGAN2', 'ORGAN3',
                    'ORGAN4', 'ORGAN5', 'ORGAN6', 'DIVATT1', 'DIVATT2', 'DIVATT3', 'DIVATT4', 'STAFFASST', 'VALIDITY',
                    'SOURCE', 'EcogPtDivatt', 'EcogPtLang', 'EcogPtMem', 'EcogPtOrgan', 'EcogPtPlan', 'EcogPtVisspat',
                    'EcogPtTotal'], save_path + 'tmpfb/clear_ECOGPT.csv')

def merge_neuro_information():

    df_gsd = pd.read_csv(save_path + 'tmpfb/clear_GDSCALE.csv', dtype=str)
    df_npi=combin_npi()

    df_neuro = pd.merge(df_npi, df_gsd, on=['RID', 'VISCODE2'], how='outer')
    df_neuro = df_neuro.drop_duplicates()
    df_neuro.to_csv(save_path + 'neuro_scale.csv', index=False)

def merge_behavior_function():

    df_fqa = pd.read_csv(save_path + 'tmpfb/clear_FAQ.csv', dtype=str)
    df_ecogpt = pd.read_csv(save_path + 'tmpfb/clear_ECOGPT.csv', dtype=str)
    df_ecogsp=get_ecogsp()

    df_ecog = pd.merge(df_ecogpt, df_ecogsp, on=['RID', 'VISCODE2'], how='outer')

    #df_fb = pd.merge(df_cdr, df_npi, on=['RID', 'VISCODE2'], how='outer')
    df_fb = pd.merge(df_fqa, df_ecog, on=['RID', 'VISCODE2'], how='outer')
    df_fb = df_fb.drop_duplicates()
    df_fb.to_csv(save_path + 'fb_scale.csv', index=False)

def main() :
    merge_base_information()
    clean_ntest()
    merge_cog_information()
    #clean_cci()
    clean_nfb()
    merge_neuro_information()
    merge_behavior_function()

if __name__ == '__main__':
    main()
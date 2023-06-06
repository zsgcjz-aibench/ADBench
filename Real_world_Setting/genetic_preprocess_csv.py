#-*- coding:utf-8 -*-

import os
import math

#file_name='./ADNI.808_indiv.minGQ_21.pass.ADNI_ID.chr'
file_name='/mnt/sdb/pwb/GeneticData/2_ADNI1_GWAS/Data/adni_gwas_v2_set'
#file_name='/mnt/sdb/pwb/GeneticData/3_ADNI_GOor2_GWAS/Data/ADNIGO2_GWAS_Set'
file_snp_name='/mnt/sdb/pwb/GeneticData/2_ADNI1_GWAS/Data/snps.txt'
save_file='/mnt/sdb/pwb/gen/ADNI1_20201020.txt'
file_N=1
isHead=True
snp_name_array=[]
snp_name=set()

genetic_data_set={}

with open(file_snp_name, 'r') as snp_name_f:
    tmp_snp_name_array = snp_name_f.readlines()
    for i in range(0,len(tmp_snp_name_array)):
        snp_name.add(tmp_snp_name_array[i].rstrip('\n').strip())
        snp_name_array.append(tmp_snp_name_array[i].rstrip('\n').strip())



dir_or_files = os.listdir(file_name)
for dir_file in dir_or_files:
    # 获取目录或者文件的路径
    if dir_file[len(dir_file)-4:len(dir_file)]=='.csv':
        dir_file_path = os.path.join(file_name, dir_file)
        user_ID=dir_file.replace('.csv','')
        print('########   '+dir_file)
        with open(dir_file_path, 'r') as t_f:
            line_array = t_f.readlines()
        for j in range(0, len(line_array)):
            t_line = line_array[j].rstrip('\n').strip()
            t_array = t_line.split(',')
            snp = t_array[4]
            sample_str = ''
            if snp in snp_name:
                gc=t_array[13]
                af1=t_array[7]
                af2=t_array[8]
                if af1!='-' and af2!='-':
                    sample_str = af1 + ','+af2+','+gc

                    if user_ID not in genetic_data_set:
                        genetic_data_set[user_ID] = {}
                    sample_value_dic = genetic_data_set[user_ID]
                    if snp in sample_value_dic:
                        print("########### Repeated SNP:     " + user_ID + ',' + snp)
                    else:
                        sample_value_dic[snp] = sample_str


dataset_str='RID'
for i in range(0,len(snp_name_array)):
    dataset_str=dataset_str+','+snp_name_array[i]+'_A,'+snp_name_array[i]+'_G,'\
                +snp_name_array[i]+'_C,'+snp_name_array[i]+'_T,'+snp_name_array[i]+'_Confidence'
dataset_str=dataset_str+'\n'

for k in genetic_data_set:
    sample_str=str(int(k.split('_')[2]))
    tmp_sample=genetic_data_set[k]
    for i in range(0,len(snp_name_array)):
        if snp_name_array[i] in tmp_sample:
            tmp_snp=tmp_sample[snp_name_array[i]].split(',')
            #print('******')
            #print(tmp_snp)
            dna_base=['0','0','0','0']
            for j in range(0,len(tmp_snp)-1):
                if tmp_snp[j]=='A':
                    dna_base[0]='1'
                elif tmp_snp[j]=='G':
                    dna_base[1]='1'
                elif tmp_snp[j]=='C':
                    dna_base[2]='1'
                elif tmp_snp[j]=='T':
                    dna_base[3]='1'
            #print(dna_base)
            rate=float(tmp_snp[-1])
            if rate>1.0:
                rate=1-(math.pow(10,-(rate/10.0)))
            elif rate<0:
                rate=-4
            tmp_snp_str=dna_base[0]+','+dna_base[1]+','+dna_base[2]+','+dna_base[3]+','+str(rate)

            sample_str=sample_str+','+tmp_snp_str
        else:
            tmp_snp=''
            sample_str = sample_str+',,,,,'
    sample_str = sample_str +'\n'
    dataset_str=dataset_str+sample_str

with open(save_file, 'a') as f:
    f.write(dataset_str)









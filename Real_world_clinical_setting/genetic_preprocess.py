#-*- coding:utf-8 -*-
import math

#碱基的排序AGCT
file_name='/mnt/sdb/pwb/GeneticData/1_ADNI_WGS+Omni2.5M/Data/snp-gatk/ADNI.808_indiv.minGQ_21.pass.ADNI_ID.chr'
#file_name='/mnt/sdb/pwb/GeneticData/8_adni3/PLINK_Final/adni3'
#file_name='./test'
file_snp_name='/mnt/sdb/pwb/GeneticData/2_ADNI1_GWAS/Data/snps.txt'
save_file='/mnt/sdb/pwb/gen/save_20200819.txt'
file_N=23
isHead=True
snp_name_array=[]
snp_name=set()

genetic_data_set={}

with open(file_snp_name, 'r') as snp_name_f:
    tmp_snp_name_array = snp_name_f.readlines()
    for i in range(0,len(tmp_snp_name_array)):
        snp_name.add(tmp_snp_name_array[i].rstrip('\n'))
        snp_name_array.append(tmp_snp_name_array[i].rstrip('\n'))

for i in range(0,file_N):
    print('Processing file index : '+str(i+1))
    t_file_name=file_name+str(i+1)+'.vcf'
    #t_file_name=file_name+'.vcf'
    title_item=[]
    with open(t_file_name,'r') as t_f:
        line_array=t_f.readlines()
    for j in range(0,len(line_array)):
        if j%2000000==0:
            print('File index '+str(i+1)+'   row index: '+str(j))
        if line_array[j][0:2]!='##':
            if line_array[j][0:1]=='#':
                print('#################        '+line_array[j])
                t_line=line_array[j].rstrip('\n').replace('#','')
                t_array=t_line.split('\t')
                title_item.extend(t_array)
            else:
                t_line = line_array[j].rstrip('\n')
                t_array = t_line.split('\t')

                chrom=t_array[0]
                pos=t_array[1]
                id=t_array[2]
                ref=t_array[3]
                alt=t_array[4]
                quat=t_array[5]
                filter_g=t_array[6]
                info=t_array[7]
                format_g=t_array[8]

                if id in snp_name:
                    #ref_array=ref.split(',')
                    alt_array=alt.split(',')

                    format_g_array=format_g.split(':')
                    format_g_dic={}
                    for l in range(0,len(format_g_array)):
                        format_g_dic[format_g_array[l]]=l
                    if 'GT' in format_g_dic:
                        for k in range(9,len(t_array)):
                            sample_str=''
                            value_array=t_array[k].split(':')
                            gt=value_array[format_g_dic['GT']]
                            if gt!='./.':
                                gt_array=gt.split('/')
                                for m in range(0,len(gt_array)):
                                    index_gt=int(gt_array[m])
                                    if index_gt==0:
                                       sample_str=sample_str+ref+','
                                    else:
                                        sample_str = sample_str + alt_array[index_gt-1] + ','
                                if 'GQ' in format_g_dic:
                                    sample_str=sample_str+value_array[format_g_dic['GQ']]
                                else:
                                    sample_str = sample_str + '-1'
                                if title_item[k] not in genetic_data_set:
                                    genetic_data_set[title_item[k]]={}
                                sample_value_dic=genetic_data_set[title_item[k]]
                                if id in sample_value_dic:
                                    print("########### Repeated SNP:     "+title_item[k]+','+id)
                                else:
                                    sample_value_dic[id]=sample_str


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











#-*- coding:utf-8 -*-
import os

order_list=[]
id2path=[]

#file_list=[]
#save_path_MRI='/mnt/sdc/ADNI_IMAGE_20201007/c_MRI/'
#save_path_PET='/mnt/sdc/ADNI_IMAGE_20201007/c_PET/'


#MRI_path='/mnt/sdc/ADNI_IMAGE_20201007/MRI'
#PET_path='/mnt/sdc/ADNI_IMAGE_20201007/PET'

dataset_path='/data/huangyunyou/ADNI/'
#/data/huangyunyou/ADNI/

save_path_MRI='IMAGE/c_MRI/'
save_path_PET='IMAGE/c_PET/'


MRI_path='IMAGE/MRI'
PET_path='IMAGE/PET'

# convert .dcm to .nii and rename the file by image ID
# dcm2niix -o /mnt/sdc/ADNI_IMAGE_20201007/testx -f I609338.nii ./I609338
def get_file_path(root_path,save_path,pre_save_path='',pre_or_path=''):
    # 获取该目录下所有的文件名称和目录名称
    dir_or_files = os.listdir(root_path)

    for dir_file in dir_or_files:
        # 获取目录或者文件的路径
        dir_file_path = os.path.join(root_path, dir_file)
        # 判断该路径为文件还是路径
        if os.path.isdir(dir_file_path):
            # 递归获取所有文件和目录的路径
            get_file_path(dir_file_path,save_path,pre_save_path,pre_or_path)
        else:
            if '.dcm' in dir_file:
                fileArray = dir_file.split('_')
                fileID = fileArray[len(fileArray) - 1]
                fileID = fileID.split('.')[0]
                imageID = fileID[1:len(fileID)]
                print('Converting image : '+imageID)

                tmp_order='dcm2niix -o '+save_path+' -f '+imageID+' '+root_path

                tmp_map=imageID+','+save_path.replace(pre_save_path,'')+imageID+'.nii'

                order_list.append(tmp_order)
                id2path.append(tmp_map)
                break

            elif '.nii' in dir_file:

                fileArray = dir_file.split('_')
                fileID = fileArray[len(fileArray) - 1]
                fileID = fileID.split('.')[0]
                imageID = fileID[1: len(fileID)]
                print('Moving image : ' + imageID)

                tmp_map = imageID + ',' + dir_file_path.replace(pre_or_path,'')
                #tmp_order='cp '+dir_file_path+' '+save_path+dir_file
                #order_list.append(tmp_order)
                id2path.append(tmp_map)

            #break
                #tmp_file=''+root_path+'_'+root_path+'.nii'
                #file_list
    #return file_list

#dcm2niix -z n -p y -f test.nii -o /mnt/sdc/mni/mertest/003_S_1122/ADNI3_FDG__AC_/2017-07-28_09_26_02.0 /mnt/sdc/mni/mertest/003_S_1122/ADNI3_FDG__AC_/2017-07-28_09_26_02.0/I942968
#order_list=[]
#file_list=[]
get_file_path(dataset_path+MRI_path,dataset_path+save_path_MRI,dataset_path,dataset_path)

get_file_path(dataset_path+PET_path,dataset_path+save_path_PET,dataset_path,dataset_path)

str_command=''
for str_value in order_list:
    str_command=str_command+str_value+'\n'

str_command_map=''
for str_value in id2path:
    str_command_map=str_command_map+str_value+'\n'


save_file='/data/huangyunyou/ADNI/src/dcm2nii.sh'
path=os.path.dirname(save_file)
if not os.path.exists(path):
    os.makedirs(path)
with open(save_file, 'a') as f:
    f.write(str_command)


save_file_map='/data/huangyunyou/ADNI/IMAGE/id2imagePath.txt'
path_map=os.path.dirname(save_file_map)
if not os.path.exists(path_map):
    os.makedirs(path_map)
with open(save_file_map, 'a') as f:
    f.write(str_command_map)

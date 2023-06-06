import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from glob import glob
from subprocess import call
import os
from glob import glob
import json
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import datetime

thread_num = 1


def read_json(file):
    with open(file, 'r') as f:
        return json.load(f)


def getID(filename):
    if '.nii' in filename:
        return filename.strip('.nii')  
    elif '.jpg' in filename:
        return filename.strip('.jpg')


def create_folder(folder):
    if not os.path.exists(folder):
        os.mkdir(folder)


def show_scan(path, filename, output_folder, mode_flag):
    image_name = filename.replace('nii', 'jpg')
    data = nib.load(path + '/' + filename).get_data()
    plt.subplot(1, 3, 1)
    plt.imshow(data[95, :, :])
    plt.colorbar()
    plt.subplot(1, 3, 2)
    plt.imshow(data[:, 100, :])
    plt.colorbar()
    plt.subplot(1, 3, 3)
    plt.imshow(data[:, :, 100])
    plt.colorbar()
    plt.savefig(output_folder + '/' + mode_flag + '/' + image_name)
    plt.close()


def find_cases(folder, extention):
    hashmap = {}
    for root, dirs, files in os.walk(folder, topdown=False):
        print("root:", root)
        print("dirs:", dirs)
        for file in files:
            print("file:", file)
            if file.endswith(extention):
                print("file:", file)
                hashmap[getID(file)] = os.path.join(root, file)
    return hashmap

#multithreading
def main_mul(dir, root_folder, step_7_f, step_7_g):
    """
    
    :param dir: 
    :param root_folder: 
    :param step_7_f: 
    :param step_7_g: 
    :return: 
    """
    str_dir = " ".join(dir)
    os.system("./pipeline.sh '" + str_dir + "' " + root_folder + ' ' + step_7_f + ' ' + step_7_g)


def zip_main_mul(args):
    main_mul(args[0], args[1], args[2], args[3])


if __name__ == "__main__":
    csv_path = r"./test_data/demo.txt" #txt path containing the file name to be processed
    config = read_json('pipeline_config.json') #rootPath: the location of the target file; rawPath: the location where the original nii file is stored
    file_path_arr = []
    with open(csv_path, 'r') as f:
        for line in f.readlines():
            file_path_arr.append(line.strip('\n'))
    print(file_path_arr)
    len_of_arr = len(file_path_arr)
    thread_count = int(len_of_arr / thread_num)
    thread_path_arr = []
    for i in range(thread_num):
        thread_path_arr.append(
            (file_path_arr[i * thread_count:(i + 1) * thread_count], config['rootPath'], config['step7F'],
             config['step7G']))
    if len_of_arr % thread_num != 0:
        thread_path_arr.append((file_path_arr[thread_count * thread_num:len_of_arr], config['rootPath'],
                                config['step7F'], config['step7G']))
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    with ThreadPoolExecutor(max_workers=thread_num) as pool:
        results = pool.map(zip_main_mul, thread_path_arr)
        for r in results:
            continue
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))



"""
workflow:
1. run the main.py with default pipeline_config.json to process all cases. processed nifti will be saved in scans/; 
   images will be ploted in images/ for the inspection of processing quality. 
2. review the cases by looking through the images/ folder, put the images you think that are good enough to the good/ folder. 
   the cases in the good/ folder will never be processed again.
3. while there are bad cases left, identify common problems, like (a) extra skull left; (b) grey matter being cut by BET, or other problems
4. create a folder (folder name to indicate the problem type) and put the cases's images with similar problem into this folder 
6. change the "mode" in pipeline_config.json to be the same as the new folder name, 
   adjust the parameters in the pipeline_config.json to address the problems of this type of problem.
5. run the main.py again to batch-wise process all cases with this common problem
6. review the outcome from the images/ folder, put good cases in good/ folder
7. loop back to step 4 until all cases are in the good/ folder

folder structure:
rootPath/
    scans/               save the most updated nifit scans 
    images/              save the most updated image 
    tmp/                 tmp working folder
    good/                a pool of properly processed cases
    npy/                 save the most updated npy scans
"""

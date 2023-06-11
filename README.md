#  "ADBench: A Multi-setting Benchmark Suite for Evaluating Alzheimer’s Disease Diagnostic Models"


This is the official implementation of our conference paper : "ADBench: A Multi-setting Benchmark Suite for Evaluating Alzheimer’s Disease Diagnostic Models".

## 1. Introduction

The **ADBench** is a comprehensive benchmark for Alzheimer’s disease (AD) diagnosis, which contains 10 diverse datasets, covers four clinical settings: a closed setting for classification, a data silos setting for federated learning (FL), an open setting for open set identification, and a real-world setting for implementation, and systematic evaluation with highlight the challenges in the development of AD diagnostic models. 
![image](https://github.com/zsgcjz-aibench/ADBench/blob/main/images/Figure1.jpg)

## 2. Create issues from this repository
Please contact us at `huangyunyou@gxnu.edu.cn`. We will reply the issue within 14 days.

## 3. Data Sources 
All the experimental data is from [[Alzheimer’s Disease Neuroimaging Initiative (ADNI)]](https://adni.loni.usc.edu/). Researchers are able to download experimental data through the [[Image & Data Archive]](https://ida.loni.usc.edu/login.jsp). Since ADNI prohibits distributing data, researchers need to independently download Study Data, Genetic Data, MRI and PET in ADNI before May 2, 2019.

## 4. Environment
Ubuntu 20.04.3 LTS  
Python 3.8.10  
CUDA 11.1  
cuDNN 8  
NVIDIA GeForce RTX 3090 GPU  
Intel(R) Core(TM) i9-10900X CPU @ 3.70GHz

## 5. Real-world Clinical Setting

### 
- [Demo](#5)
- [Data Preparation](#1)
- [Usage](#2)
  - [Training&Tesing](#3)
  - [Evaluation](#4)
- [License](#7)
- [Citation](#6)
  <!-- - [Visualization](#5) -->
  
<span id="5"></span>
### Demo
- Please clone our environment using the following command:
  ```
  pip install -r requirements.txt
  ```
- Requirement
  ```
  Ubuntu 20.04.3 LTS  
  Python 3.8.10  
  CUDA 11.1  
  cuDNN 8  
  ```
<span id="1"></span>
### Data Preparation
- All of the data involving AD used in this work are obtained from Alzheimer’s Disease Neuroimaging Initiative (ADNI) [database](http://adni.loni.usc.edu). ADNI is licensed under ADNI Data Sharing and Publications Committee (DPC) according to [link](https://adni.loni.usc.edu/data-samples/access-data/). The personally identifiable information of all subjects in ADNI has been removed, and access can be applied directly through [link](https://ida.loni.usc.edu/explore/jsp/register/register.jsp).
- The data should be organized like:

```
├── datasets
│  ├── images
│  │  ├── NIFTI
│  │  ├── NPY
|  ├── csv
│  │  ├── Open_clinical_setting
│  │  │  ├──train & test & val
│  │  ├── Data_silos_clinical_setting
│  │  │  ├──train & test & val
│  │  ├── Closed_clinical_settig
│  │  │  ├──train & test & val
│  │  ├── Real_world_clinical_setting
│  │  │  ├──train & test & val
     
```
- We provide the preprocessing code for the data in the `datasets/images/MRI_process/`
- The image data is registered using [FSL](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/), and the template used is "`MNI152_T1_1mm_brain`"
- We have provided some sample data（[.npy](https://drive.google.com/file/d/12lEgIub6i65M4QEDNRKp4eItgxZQmHPd/view?usp=sharing) and [.nii](https://drive.google.com/file/d/1ucTAlAEm-J3qLoReLWLMwvJWdKNl_iBe/view?usp=sharing)） that can be downloaded from Google Drive

<span id="2"></span>
### Usage
<span id="3"></span>
- #### Training & Tesing
Each scene contains multiple models, and the execution commands for each model may be different. Listed below are just some of them.

#  Real-world Clinical Setting

Step 1: Adjust the file path in the code according to the data storage location.

Step 2: Dataset Preparation.

```
python3 scale.py

python3 biospecimen.py

python3 genetic_preprocess_csv.py

python3 genetic_preprocess.py

python3 dcm2nii.py

sudo chmod 777 dcm2nii.sh

./dcm2nii.sh


#Crate training set(ac_train.tfrecord), validation set(ac_eval.tfrecord), and test set (ac_test.tfrecord and mci_test.tfrecord) of AD, CN and MCI.
python3 create_data_set_2v.py 

#Crate test set of SMC(smc_test.tfrecord).
python3 create_data_set_8v.py 
     
```
Step 3: Traning and tesing model of [[42]](https://www.nature.com/articles/s41467-022-31037-5)

```
# Save the model ae_ac_0.0005_32_210v.h5 for evaluation of [6] and [39]
python3 OpenClinicalAI.py

```
Step 4: Traning and tesing model of [[6]](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Bendale_Towards_Open_Set_CVPR_2016_paper.pdf) and [[39]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Perera_Generative-Discriminative_Feature_Representations_for_Open-Set_Recognition_CVPR_2020_paper.pdf)

```
python3 Thr_OpenMax.py

python3 Thr_OpenMAX_Performance.py

```



- #### Evaluation
<span id="4"></span>
```
## For Model 1
python ###.py
     
```

```
## For Model 2
python ###.py
     
```

<span id="7"></span>
### License
Distributed under the MIT License. See LICENSE for more information.

<span id="6"></span>
### Citation
If you use this work for your research, please cite our paper:
```
paper cite here
     
```

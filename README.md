#  "ADBench: A Multi-setting Benchmark Suite for Evaluating Alzheimer’s Disease Diagnostic Models"


This is the official implementation of our conference paper : "ADBench: A Multi-setting Benchmark Suite for Evaluating Alzheimer’s Disease Diagnostic Models".

## Introduction

The paper discusses the importance of correctly diagnosing Alzheimer's disease (AD), which can greatly reduce the burden on society and individuals. Existing AD diagnosis works have limitations as they assume an ideal closed setting where all categories are known prior and the diagnostic strategy is the same for all subjects. To address this limitation, the authors present a multi-setting benchmark suite called ADBench for evaluating AD diagnostic models. ADBench contains ten benchmark datasets covering four clinical settings: a closed setting for classification, a data silos setting for federated learning (FL), an open setting for open set identification, and a real-world setting for implementation. The benchmark suite contains 13 categories of common AD diagnosis data from 67 separate sites and provides a joint simulation of subjects and medical institutions in the real-world setting. The authors evaluate AD diagnostic models in four settings of research and show that the performance of mainstream AD diagnosis models drops sharply in the open and real-world settings. They also provide a quantitative and qualitative analysis of their results, highlighting challenges in the development of AD diagnostic models, and suggest that clinical scale data can improve the robustness of the models. The authors conclude that their benchmark suite can promote the innovation and clinical implementation of AD diagnostic models in different research domains. 
![image](https://github.com/Only-Child/ADBench/blob/main/resources/framwork_1.png)

## Create issues from this repository
Please contact us at `huangyunyou@gxnu.edu.cn`. We will reply the issue within 14 days.

### Tab of Content
- [Demo](#5)
- [Data Preparation](#1)
- [Usage](#2)
  - [Training](#3)
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
- We provide the preprocessing code for the data in the `resources/`
- The image data is registered using [FSL](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/), and the template used is "MNI152_T1_1mm_brain"
<!-- - We have provided some sample data that can be downloaded from [Google Drive](https://drive.google.com/file/d/1PiTzGQEVV7NO4nPaHeQv61WgDxoD76nL/view?usp=share_link) . -->

<span id="2"></span>
### Usage
<span id="3"></span>
- #### Training
Each scene contains multiple models, and the execution commands for each model may be different. Listed below are just some of them.

```
## For Real-world Clinical Setting

Step 1: Adjust the file path in the code according to the data storage location.

Step 2: Dataset Preparation.

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


```
## For Model 2
python ###.py
     
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

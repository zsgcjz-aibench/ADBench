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
```
Ubuntu 20.04.3 LTS  
Python 3.8.10  
CUDA 11.1  
cuDNN 8  
NVIDIA GeForce RTX 3090 GPU  
Intel(R) Core(TM) i9-10900X CPU @ 3.70GHz
```
    
## 5. Real-world Clinical Setting
- Step 1: Adjust the file path in the code according to the data storage location.

- Step 2: Dataset Preparation.

```
python3 scale.py

python3 biospecimen.py

python3 genetic_preprocess_csv.py

python3 genetic_preprocess.py

python3 dcm2nii.py

sudo chmod 777 dcm2nii.sh

./dcm2nii.sh

# Crate training set(ac_train.tfrecord), validation set(ac_eval.tfrecord), and test set (ac_test.tfrecord and mci_test.tfrecord) of AD, CN and MCI.
python3 create_data_set_2v.py 

# Crate test set of SMC(smc_test.tfrecord).
python3 create_data_set_8v.py
```

- Step 3: Traning and tesing model of [[42]](https://www.nature.com/articles/s41467-022-31037-5).
```
# Save the model ae_ac_0.0005_32_210v.h5 for evaluation of [6] and [39].
python3 OpenClinicalAI.py
```
- Step 4: Traning and tesing model of [[6]](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Bendale_Towards_Open_Set_CVPR_2016_paper.pdf) and [[39]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Perera_Generative-Discriminative_Feature_Representations_for_Open-Set_Recognition_CVPR_2020_paper.pdf).
```
python3 Thr_OpenMax.py

python3 Thr_OpenMAX_Performance.py
```
## 6. Open Clinical Setting

- Environment
```
python 3.9.13
pandas 1.4.4
torch 1.10.2
path 16.6.0
numpy 1.21.5
tqdm 4.64.1
```
- Step 1: Adjust the file path in the code according to the data storage location.

- Step 2: Dataset Preparation(need FSL to be installed).
    
    We provided this bash pipeline (datasets/images/MRI_preprocess/pipeline.sh) to perform this step. To run the registration.sh on a single case:
```

bash pipeline.sh folder_of_raw_nifti/ imagename.nii output_folder_for_processed_data/

```
   To register all data in a folder, you can use the python script 
    
```
python pipeline_mul.py
```

clip out the intensity outliers and Bias field correction
```
python nifti_to_numpy.py

python biasFieldCorrection.py
```

- Step 3: Train, validate and test a model.

    The modelname.py contains the interfaces for initializing, training, validating, testing, saving, loading the model. See below for a basic example usage.
```
# Need to adjust to the corresponding model name
python DSA_openmax.py 

# python Dynamic_openmax.py
# python CNN_MLP_openmax.py 
# python LRP_openmax.py
# python FCN_model.py
# python MLP_openmax.py 
```

## 7. Data Silos Clinical Setting
- Step 1: Adjust the file path in the code according to the data storage location.

- Step 2: Dataset Preparation.
```
    # Same as data preprocessing in Open Clinical Setting.
```
- Step 3: Train, validate a model.
 The train_modelname.py contains the interfaces for initializing, training, testing, saving the model. See below for a basic example usage.
```
# randomly divide the training set into multiple disjoint clients.
# Need to adjust to the corresponding model name
python train_CNN_MLP.py # [42] model of paper

# python train_LRP.py # [10] model of paper
# python train_Vox.py # [28] model of paper
# python train_Dynamic.py # [55] model of paper
# python train_DSA.py # [21] model of paper
# python train_FCN.py # [41] model of paper
# python train_MLP_A.py # [41] model of paper
```
- Step 4: Evaluate a model. See below for a basic example usage.
```
# randomly divide the training set into multiple disjoint clients. 
# Need to adjust to the corresponding model name
python test_model_CNN_MLP.py # [42] model of paper

# python test_model_LRP.py # [10] model of paper
# python test_model_Vox.py # [28] model of paper
# python test_model_Dynamic.py # [55] model of paper
# python test_model_DSA.py # [21] model of paper
# python test_model_MLP_A.py # [41] model of paper
```

## 8. Closed Setting
- Step 1: Adjust the file path in the code according to the data storage location.

- Step 2: Dataset Preparation.
``` 
    # Same as data preprocessing in Open Clinical Setting.
``` 
- Step 3: Train, validate a model.
 The modelname.py contains the interfaces for initializing, training, testing, saving the model. See below for a basic example usage.
```
# randomly divide the training set, validation set and test set.
# Need to adjust to the corresponding model name
python train_CNN_MLP.py # [42] model of paper

# python train_LRP.py # [10] model of paper
# python train_Vox.py # [28] model of paper
# python train_Dynamic.py # [55] model of paper
# python train_DSA.py # [21] model of paper
# python train_FCN.py # [41] model of paper
# python train_MLP_A.py # [41] model of paper
```
- Step 4: Evaluate a model. See below for a basic example usage.
```
# randomly divide the training set into multiple disjoint clients. 
# Need to adjust to the corresponding model name
python test_model_CNN_MLP.py # [42] model of paper

# python test_model_LRP.py # [10] model of paper
# python test_model_Vox.py # [28] model of paper
# python test_model_Dynamic.py # [55] model of paper
# python test_model_DSA.py # [21] model of paper
# python test_model_MLP_A.py # [41] model of paper
```

## 9. License
- Our codes were released with MIT License. 

import pandas as pd
import numpy as np
from eval_index_cal_loss import *

G_base_cog_train = r'C:\Users\hp\Desktop\model_G\base_cog_train.csv'
G_base_cog_test = r'C:\Users\hp\Desktop\model_G\base_cog_test.csv'
G_base_cog_mri_train = r'C:\Users\hp\Desktop\model_G\base_cog_mri_train.csv'
G_base_cog_mri_test = r'C:\Users\hp\Desktop\model_G\base_cog_mri_test.csv'
G_open_train = r'C:\Users\hp\Desktop\model_G\GAN4_train_strategy_72_filter_result\GAN4_train_strategy_72_filter_result\ac_train.csv'
G_open_ac_test = r'C:\Users\hp\Desktop\model_G\GAN4_train_strategy_72_filter_result\GAN4_train_strategy_72_filter_result\ac_test.csv'
G_open_mci_test = r'C:\Users\hp\Desktop\model_G\GAN4_train_strategy_72_filter_result\GAN4_train_strategy_72_filter_result\mci_test.csv'
G_open_smc_test = r'C:\Users\hp\Desktop\model_G\GAN4_train_strategy_72_filter_result\GAN4_train_strategy_72_filter_result\smc_test.csv'

G_base_cog_train_data = pd.read_csv(G_base_cog_train, header=0, index_col=0)
G_base_cog_test_data = pd.read_csv(G_base_cog_test, header=0, index_col=0)
G_base_cog_mri_train_data = pd.read_csv(G_base_cog_mri_train, header=0, index_col=0)
G_base_cog_mri_test_data = pd.read_csv(G_base_cog_mri_test, header=0, index_col=0)
G_open_train_data = pd.read_csv(G_open_train, header=0)
G_open_ac_test_data = pd.read_csv(G_open_ac_test, header=0)
G_open_mci_test_data = pd.read_csv(G_open_mci_test, header=0)
G_open_smc_test_data = pd.read_csv(G_open_smc_test, header=0)


data_select = 2

# G_base_cog
if data_select == 1:
    # train imformation loss loss_thr
    G_base_cog_train_loss = np.array(G_base_cog_train_data)[:, 4]
    G_base_cog_thr_loss = np.percentile(G_base_cog_train_loss, 70)
    # test information av loss label
    G_base_cog_test_av = np.array([x for i in np.array(G_base_cog_test_data)[:, 2] for x in str(i).strip('[]').rstrip().split()])
    G_base_cog_test_av = np.array([float(x) for x in G_base_cog_test_av]).reshape(-1, 2)
    G_base_cog_test_loss = np.array(G_base_cog_test_data)[:, 4]
    print('--------G_base_cog_test_loss------------')
    print(G_base_cog_test_loss)
    G_base_cog_test_label = np.array(G_base_cog_test_data)[:, 5]
    print('--------G_base_cog_test_label------------')
    print(G_base_cog_test_label)

    cal_CI_plot(G_base_cog_test_av, G_base_cog_test_label, G_base_cog_test_loss, G_base_cog_thr_loss)

elif data_select == 2:
    # train imformation loss loss_thr
    G_base_cog_mri_train_loss = np.array(G_base_cog_mri_train_data)[:, 4]
    G_base_cog_mri_thr_loss = np.percentile(G_base_cog_mri_train_loss, 70)
    # test information av loss label
    G_base_cog_mri_test_av = np.array([x for i in np.array(G_base_cog_mri_test_data)[:, 2] for x in str(i).strip('[]').rstrip().split()])
    G_base_cog_mri_test_av = np.array([float(x) for x in G_base_cog_mri_test_av]).reshape(-1, 2)
    G_base_cog_mri_test_loss = np.array(G_base_cog_mri_test_data)[:, 4]
    print('--------G_base_cog_mri_test_loss------------')
    print(G_base_cog_mri_test_loss)
    G_base_cog_mri_test_label = np.array(G_base_cog_mri_test_data)[:, 5]
    print('--------G_base_cog_mri_test_label------------')
    print(G_base_cog_mri_test_label)

    cal_CI_plot(G_base_cog_mri_test_av, G_base_cog_mri_test_label, G_base_cog_mri_test_loss, G_base_cog_mri_thr_loss)


elif data_select == 3:
    G_open_train_loss = np.array(G_open_train_data)[:, 6]
    G_open_thr_loss = np.percentile(G_open_train_loss, 90)
    # test av loss label
    G_open_ac_test_av = np.array([np.array(G_open_ac_test_data)[:, 2],np.array(G_open_ac_test_data)[:, 3]]).T
    G_open_mci_test_av = np.array([np.array(G_open_mci_test_data)[:, 2], np.array(G_open_mci_test_data)[:, 3]]).T
    G_open_smc_test_av = np.array([np.array(G_open_smc_test_data)[:, 2], np.array(G_open_smc_test_data)[:, 3]]).T
    G_open_test_av = np.concatenate((G_open_ac_test_av, G_open_mci_test_av, G_open_smc_test_av), axis=0)

    G_open_ac_test_loss = np.array(G_open_ac_test_data)[:, 6]
    G_open_mci_test_loss = np.array(G_open_mci_test_data)[:, 6]
    G_open_smc_test_loss = np.array(G_open_smc_test_data)[:, 6]
    G_open_test_loss = np.concatenate((G_open_ac_test_loss, G_open_mci_test_loss, G_open_smc_test_loss))

    G_open_ac_test_label = np.array(G_open_ac_test_data)[:, 8]
    G_open_mci_test_label = np.array(G_open_mci_test_data)[:, 8]
    G_open_smc_test_label = np.array(G_open_smc_test_data)[:, 8]
    G_open_test_label = np.concatenate((G_open_ac_test_label, G_open_mci_test_label, G_open_smc_test_label))

    cal_CI_plot(G_open_test_av, G_open_test_label, G_open_test_loss, G_open_thr_loss)



















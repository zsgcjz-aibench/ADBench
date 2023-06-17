import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc,accuracy_score
import random
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt



def countX(lst, x):
    count = 0
    for label in lst:
        if label == x:
            count += 1
    return count



def correct_pred_count(preds, labels):
    labels = labels
    preds = preds
    un_pcount = 0
    ad_pcount = 0
    cn_pcount = 0
    for index, pred in enumerate(preds):
        if np.amax(pred) >= 0.95:   
            if np.amax(pred) == pred[0]:
                if labels[index] == 0:  
                    cn_pcount += 1
            if np.amax(pred) == pred[1]:  
                if labels[index] == 1:  
                    ad_pcount += 1
        else: 
            if labels[index] == 2: 
                un_pcount += 1

    print("cn_pcount= ", cn_pcount, "ad_pcount= ", ad_pcount, "un_pcount= ", un_pcount)

    return cn_pcount, ad_pcount, un_pcount



def cal_acc(preds, labels):
    cn_pcount, ad_pcount, un_pcount = correct_pred_count(preds, labels)

    accuracy = (cn_pcount + ad_pcount + un_pcount) / len(labels)

    return accuracy



def cal_sensitivity(preds, labels):
    cn_pcount, ad_pcount, un_pcount = correct_pred_count(preds, labels)

    print("cn count= ", countX(labels, 0))
    print("ad count= ", countX(labels, 1))
    print("un count= ", countX(labels, 2))


    cn_sen = cn_pcount / countX(labels, 0)
    ad_sen = ad_pcount / countX(labels, 1)
    un_sen = un_pcount / countX(labels, 2)

    return cn_sen, ad_sen, un_sen


#================2_class===========
def sensitivity2(y_test, y_hat):
    positive_num = list(y_test).count(1)
    TP = 0
    if positive_num != 0:
        for i in range(len(y_hat)):
            if (y_test[i] == 1) & (y_hat[i] > 0.5):
                TP += 1
        # print("positive_num= ", positive_num, "(TP + FN)= ", (TP + FN))
        # print("TP= ", TP, "positive_num= ", positive_num)
        return TP / positive_num
    else:
        return 0


def specificity2 (y_test, y_hat):
    negitive_num = list(y_test).count(0)
    TN = 0
    if negitive_num != 0:
        for i in range(len(y_hat)):
            if (y_test[i] == 0) & (y_hat[i] <= 0.5):
                TN += 1
        return TN / negitive_num
    else:
        return 0



def cal_roc_curve(pred, test_y):
    fpr, tpr, thresholds = roc_curve(test_y, pred)  
    roc_auc = auc(fpr, tpr) 

    return roc_auc



def ont_hot_code(labels):
    # cn_code = 0
    # ad_code = 1
    # un_code = 2
    labels = np.array(labels)
    ohe = OneHotEncoder()
    ohe.fit([[0], [1], [2]])
    code_df = ohe.transform(labels.reshape(-1,1)).toarray()

    cn_label = code_df[:, 0] #narray
    ad_label = code_df[:, 1]
    un_label = code_df[:, 2]

    return cn_label, ad_label, un_label


def ont_hot_code_close(labels):
    # cn_code = 0
    # ad_code = 1
    labels = np.array(labels)
    ohe = OneHotEncoder()
    ohe.fit([[0], [1]])
    code_df = ohe.transform(labels.reshape(-1,1)).toarray()

    cn_label = code_df[:, 0] #narray
    ad_label = code_df[:, 1]

    return cn_label, ad_label





def get_score_final(ys_true, openmax_rate=[],
                    thr=0.75):  
    nb_outputs = len(ys_true)  
    ys_pred_open_max = []

    if len(openmax_rate) > 0:  # = 3
        for i in range(nb_outputs):
            ys_pred_open_max.append([])  # ys_pred_open_max= [[], [] , []]

    for j in range(len(openmax_rate)):  # len(openmax_rate) = 3
        tmp_open_max = openmax_rate[j]  # openmax_rate=[ad_pred, cn_pred , un_pred]三维数组（n, 3)
        index = np.argmax(tmp_open_max)  

        thr_low = False
        for l in range(nb_outputs):  # = 3
            if index == (nb_outputs - 1):  # index =? 2
                if l == index:
                    ys_pred_open_max[l].append(1.0)
                else:
                    ys_pred_open_max[l].append(0.0)
            else:
                if l == index:
                    if tmp_open_max[l] >= thr:
                        ys_pred_open_max[l].append(1.0)
                        thr_low = True
                    else:
                        ys_pred_open_max[l].append(0.0)
                else:
                    if l == (nb_outputs - 1):
                        if thr_low:
                            ys_pred_open_max[l].append(0.0)
                        else:
                            ys_pred_open_max[l].append(1.0)
                    else:
                        ys_pred_open_max[l].append(0.0)

    retValue = []

    for i in range(0, nb_outputs):
        c_true = ys_true[i]
        c_pred = ys_pred_open_max[i]

        sensitivity_c = 0
        fpr = 0
        specificity_c = 0
        f_unkown = 0

        for k in range(len(c_pred)):
            true_value = c_true[k]
            pred_value = c_pred[k]

            if true_value == 1 and pred_value == 1:
                sensitivity_c += 1

            if (i == 0 or i == 1) and true_value == 1 and pred_value == 0 and ys_pred_open_max[2][k] == 1:
                f_unkown += 1

            if true_value == 0 and pred_value == 1:
                fpr += 1

            if true_value == 0 and pred_value == 0:
                specificity_c += 1

        retValue.append([sensitivity_c / c_true.sum(), specificity_c / (c_true.size - c_true.sum()), fpr / (c_true.size - c_true.sum()), f_unkown / c_true.sum()])

    acc = 0
    for i in range(0, c_true.size):
        iscorrect = True
        for j in range(nb_outputs):
            if ys_true[j][i] != ys_pred_open_max[j][i]:
                iscorrect = False
        if iscorrect:
            acc += 1

    return acc / ys_true[0].size, retValue




def get_score_final_2v(ys_true, openmax_rate=[], thr=0.5):
    nb_outputs = len(ys_true)
    ys_pred_open_max = []

    if len(openmax_rate) > 0:
        for i in range(nb_outputs):
            ys_pred_open_max.append([])

    for j in range(len(openmax_rate)):
        tmp_open_max = openmax_rate[j]
        index = np.argmax(tmp_open_max)

        thr_low = False
        for l in range(nb_outputs):
            if index == (nb_outputs - 1):
                if l == index:
                    ys_pred_open_max[l].append(1.0)
                else:
                    ys_pred_open_max[l].append(0.0)
            else:
                if l == index:
                    if tmp_open_max[l] >= thr:
                        ys_pred_open_max[l].append(1.0)
                        thr_low = True
                    else:
                        ys_pred_open_max[l].append(0.0)
                else:
                    if l == (nb_outputs - 1):
                        if thr_low:
                            ys_pred_open_max[l].append(0.0)
                        else:
                            ys_pred_open_max[l].append(1.0)
                    else:
                        ys_pred_open_max[l].append(0.0)

    retValue = []

    for i in range(0, nb_outputs):
        c_true = ys_true[i]
        c_pred = ys_pred_open_max[i]

        sensitivity_c = 0
        fpr = 0

        f_unkown = 0

        for k in range(len(c_pred)):
            true_value = c_true[k]
            pred_value = c_pred[k]

            if true_value == 1 and pred_value == 1:
                sensitivity_c += 1


            if true_value == 0 and pred_value == 1:
                fpr += 1

        retValue.append([sensitivity_c / c_true.sum(), fpr / (c_true.size - c_true.sum()), f_unkown / c_true.sum()])

    acc = 0
    for i in range(0, c_true.size):
        iscorrect = True
        for j in range(nb_outputs):
            if ys_true[j][i] != ys_pred_open_max[j][i]:
                iscorrect = False
        if iscorrect:
            acc += 1

    return acc / ys_true[0].size, retValue




def get_one_hot_2v(data_y, batch_len):  # data_y = [ad_pred, cn_pred, un_pred]
    # print(data_y)
    # print(batch_len)
    ret_value = []
    class_num = len(data_y)  # = 3
    for i in range(0, batch_len):
        tmp_one_hot = np.zeros(class_num)  # =[0,0,0]
        for j in range(class_num):  # tmp_one_hot=[0.9ad 0.1cn  0un]
            tmp_one_hot[j] = data_y[j][i]
        ret_value.append(tmp_one_hot)
    # print(ret_value)
    return ret_value

def cal_CI_plot(preds, labels):
    dataset_len = len(labels)
    cn_label, ad_label, un_label = ont_hot_code(labels)
    cn_pred = preds[:, 0]
    ad_pred = preds[:, 1]

    ad_auc_array=[]
    cn_auc_array=[]
    acc_array=[]

    ad_1_array=[]
    ad_2_array=[]
    ad_3_array=[]

    cn_1_array = []
    cn_2_array = []
    cn_3_array = []

    un_1_array = []
    un_2_array = []
    un_3_array = []

    for i in range(2000):
        sample_index = []
        for j in range(2500):
            tmp_index = random.randint(0,  dataset_len- 1)
            sample_index.append(tmp_index)
        ad_test_y_ci = ad_label[sample_index]
        ad_pred_ci = ad_pred[sample_index]
        cn_label_ci = cn_label[sample_index]
        cn_pred_ci = cn_pred[sample_index]
        un_label_ci = un_label[sample_index]

        ad_fpr, ad_tpr, thresholds = roc_curve(ad_test_y_ci, ad_pred_ci)  
        roc_auc = auc(ad_fpr, ad_tpr)  
        fpr_cn, tpr_cn, thresholds_cn = roc_curve(cn_label_ci, cn_pred_ci) 
        roc_auc_cn = auc(fpr_cn, tpr_cn)  

        acc, result = get_score_final([ad_test_y_ci, cn_label_ci, un_label_ci],get_one_hot_2v([ad_pred_ci, cn_pred_ci, np.zeros(len(cn_pred_ci))], len(cn_pred_ci)) )

        ad_auc_array.append(roc_auc)
        cn_auc_array.append(roc_auc_cn)
        acc_array.append(acc)

        ad_1_array.append(result[0][0])  #sen
        ad_2_array.append(result[0][1])  #spec
        ad_3_array.append(result[0][2]) #fpr

        cn_1_array.append(result[1][0])
        cn_2_array.append(result[1][1])
        cn_3_array.append(result[1][2])

        un_1_array.append(result[2][0])
        un_2_array.append(result[2][1])
        un_3_array.append(result[2][2])

    ad_auc_array = np.array(ad_auc_array)
    cn_auc_array = np.array(cn_auc_array)
    acc_array = np.array(acc_array)

    ad_1_array = np.array(ad_1_array)
    ad_2_array = np.array(ad_2_array)
    ad_3_array = np.array(ad_3_array)

    cn_1_array = np.array(cn_1_array)
    cn_2_array = np.array(cn_2_array)
    cn_3_array = np.array(cn_3_array)

    un_1_array = np.array(un_1_array)
    un_2_array = np.array(un_2_array)
    un_3_array = np.array(un_3_array)

    print('ad auc:  ',np.percentile(ad_auc_array,2.5),',',np.percentile(ad_auc_array,50),',',np.percentile(ad_auc_array,97.5))
    print('cn auc:  ', np.percentile(cn_auc_array, 2.5), ',', np.percentile(cn_auc_array, 50), ',',np.percentile(cn_auc_array, 97.5))
    print('acc:  ', np.percentile(acc_array, 2.5), ',', np.percentile(acc_array, 50), ',',np.percentile(acc_array, 97.5))

    print('ad_sen:  ', np.percentile(ad_1_array, 2.5), ',', np.percentile(ad_1_array, 50), ',',np.percentile(ad_1_array, 97.5))
    print('ad_spec:  ', np.percentile(ad_2_array, 2.5), ',', np.percentile(ad_2_array, 50), ',',np.percentile(ad_2_array, 97.5))
    print('ad_fpr:  ', np.percentile(ad_3_array, 2.5), ',', np.percentile(ad_3_array, 50), ',',np.percentile(ad_3_array, 97.5))

    print('cn_sen:  ', np.percentile(cn_1_array, 2.5), ',', np.percentile(cn_1_array, 50), ',',np.percentile(cn_1_array, 97.5))
    print('cn_spec:  ', np.percentile(cn_2_array, 2.5), ',', np.percentile(cn_2_array, 50), ',',np.percentile(cn_2_array, 97.5))
    print('cn_fpr:  ', np.percentile(cn_3_array, 2.5), ',', np.percentile(cn_3_array, 50), ',',np.percentile(cn_3_array, 97.5))

    print('un_sen:  ', np.percentile(un_1_array, 2.5), ',', np.percentile(un_1_array, 50), ',',np.percentile(un_1_array, 97.5))
    print('un_spec:  ', np.percentile(un_2_array, 2.5), ',', np.percentile(un_2_array, 50), ',',np.percentile(un_2_array, 97.5))
    print('un_fpr:  ', np.percentile(un_3_array, 2.5), ',', np.percentile(un_3_array, 50), ',',np.percentile(un_3_array, 97.5))


    ad_test_y = ad_label
    test_y_cn = cn_label
    test_y_unkown = un_label
    ad_pred = ad_pred
    cn_pred = cn_pred


    fpr, tpr, thresholds = roc_curve(ad_test_y, ad_pred)   



    roc_auc = auc(fpr, tpr)  

    fpr_cn, tpr_cn, thresholds_cn = roc_curve(test_y_cn, cn_pred)  



    roc_auc_cn = auc(fpr_cn, tpr_cn)  


    acc,result=get_score_final([ad_test_y,test_y_cn,test_y_unkown],get_one_hot_2v([ad_pred, cn_pred,np.zeros(len(cn_pred))], len(cn_pred)))
    #plt.scatter(x, y)
    print(acc,result)

    font1 = {'family': 'Nimbus Roman',
             'weight': 'bold',
             'style': 'normal',
             'size': 15,
             }

    font2 = {'family': 'Nimbus Roman',
             'weight': 'bold',
             'style': 'normal',
             'size': 10,
             }

    plt.scatter([result[0][1]], [result[0][0]], s=50, label='AD operating point (Sensitivity={0:.4f})'.format(result[0][0])) #ad
    plt.scatter([result[1][1]], [result[1][0]], s=50, label='CN operating point (Sensitivity={0:.4f})'.format(result[1][0])) #cn
    plt.scatter([result[2][1]], [result[2][0]], s=50, label='Unkown operating point (Sensitivity={0:.4f})'.format(result[2][0]))  # cn


    plt.plot(fpr, tpr, linewidth=1, color="red", label='AD ROC (AUC = {0:.4f})'.format(roc_auc))   
    plt.plot(fpr_cn, tpr_cn, linewidth=1, color="black", label='CN ROC (AUC = {0:.4f})'.format(roc_auc_cn))  

    plt.xlim([-0.05, 1.05])  
    plt.ylim([-0.05, 1.05])
    plt.xlabel('1-Specificity',font1)
    plt.ylabel('Sensitivity',font1)  
    plt.title('Diagnosis with Dynamic-images-VGG', fontsize=15,fontweight='bold')
    plt.legend(loc="lower right",prop=font2)
    plt.savefig('Dynamic-images-VGG_open_roc.jpg')




def cal_CI_plot_close(preds, labels):
    dataset_len = len(labels)
    cn_label, ad_label= ont_hot_code_close(labels)
    cn_pred = preds[:, 0]
    ad_pred = preds[:, 1]

    ad_auc_array=[]
    cn_auc_array=[]
    acc_array=[]

    ad_1_array=[]
    ad_2_array=[]
    ad_3_array=[]

    cn_1_array = []
    cn_2_array = []
    cn_3_array = []

    for i in range(2000):
        sample_index = []
        for j in range(2500):
            tmp_index = random.randint(0,  dataset_len- 1)
            sample_index.append(tmp_index)
        ad_test_y_ci = ad_label[sample_index]
        ad_pred_ci = ad_pred[sample_index]
        cn_label_ci = cn_label[sample_index]
        cn_pred_ci = cn_pred[sample_index]

        ad_fpr, ad_tpr, thresholds = roc_curve(ad_test_y_ci, ad_pred_ci)  
        roc_auc = auc(ad_fpr, ad_tpr)  #
        fpr_cn, tpr_cn, thresholds_cn = roc_curve(cn_label_ci, cn_pred_ci)  
        roc_auc_cn = auc(fpr_cn, tpr_cn)  

        acc, result = get_score_final_2v([ad_test_y_ci, cn_label_ci],get_one_hot_2v([ad_pred_ci, cn_pred_ci], len(cn_pred_ci)) )

        ad_auc_array.append(roc_auc)
        cn_auc_array.append(roc_auc_cn)
        acc_array.append(acc)

        ad_1_array.append(result[0][0])
        ad_2_array.append(result[0][1])
        ad_3_array.append(result[0][2])

        cn_1_array.append(result[1][0])
        cn_2_array.append(result[1][1])
        cn_3_array.append(result[1][2])


    ad_auc_array = np.array(ad_auc_array)
    cn_auc_array = np.array(cn_auc_array)
    acc_array = np.array(acc_array)

    ad_1_array = np.array(ad_1_array)
    ad_2_array = np.array(ad_2_array)
    ad_3_array = np.array(ad_3_array)

    cn_1_array = np.array(cn_1_array)
    cn_2_array = np.array(cn_2_array)
    cn_3_array = np.array(cn_3_array)


    print('ad auc:  ',np.percentile(ad_auc_array,2.5),',',np.percentile(ad_auc_array,50),',',np.percentile(ad_auc_array,97.5))
    print('cn auc:  ', np.percentile(cn_auc_array, 2.5), ',', np.percentile(cn_auc_array, 50), ',',np.percentile(cn_auc_array, 97.5))
    print('acc:  ', np.percentile(acc_array, 2.5), ',', np.percentile(acc_array, 50), ',',np.percentile(acc_array, 97.5))

    print('ad1:  ', np.percentile(ad_1_array, 2.5), ',', np.percentile(ad_1_array, 50), ',',np.percentile(ad_1_array, 97.5))
    print('ad2:  ', np.percentile(ad_2_array, 2.5), ',', np.percentile(ad_2_array, 50), ',',np.percentile(ad_2_array, 97.5))
    print('ad3:  ', np.percentile(ad_3_array, 2.5), ',', np.percentile(ad_3_array, 50), ',',np.percentile(ad_3_array, 97.5))

    print('cn1:  ', np.percentile(cn_1_array, 2.5), ',', np.percentile(cn_1_array, 50), ',',np.percentile(cn_1_array, 97.5))
    print('cn2:  ', np.percentile(cn_2_array, 2.5), ',', np.percentile(cn_2_array, 50), ',',np.percentile(cn_2_array, 97.5))
    print('cn3:  ', np.percentile(cn_3_array, 2.5), ',', np.percentile(cn_3_array, 50), ',',np.percentile(cn_3_array, 97.5))



    ad_test_y = ad_label
    test_y_cn = cn_label
    ad_pred = ad_pred
    cn_pred = cn_pred



    fpr, tpr, thresholds = roc_curve(ad_test_y, ad_pred)   

    roc_auc = auc(fpr, tpr)  

    fpr_cn, tpr_cn, thresholds_cn = roc_curve(test_y_cn, cn_pred)  



    roc_auc_cn = auc(fpr_cn, tpr_cn)  # 求auc面积


    acc,result=get_score_final_2v([ad_test_y,test_y_cn],get_one_hot_2v([ad_pred, cn_pred], len(cn_pred)))
    #plt.scatter(x, y)
    print(acc,result)

    font1 = {'family': 'Consolas',
             'weight': 'bold',
             'style': 'normal',
             'size': 11,
             }

    font2 = {'family': 'Consolas',
             'weight': 'bold',
             'style': 'normal',
             'size': 9,
             }
    plt.plot(fpr, tpr, linewidth=1, color="black", label='Dynamic-image-VGG(AUC = {0:.4f})'.format(roc_auc),
             linestyle='--')  

    plt.xlim([-0.05, 1.05])  
    plt.ylim([-0.05, 1.05])
    plt.xlabel('1-Specificity',font1)
    plt.ylabel('Sensitivity',font1)  
    plt.title('Diagnosis with Dynamic-image-VGG', fontsize=15,fontweight='bold')
    plt.legend(loc="lower right",prop=font2)
    plt.savefig('Dynamic-image-VGG_close_roc.jpg')




def cal_CI_plot_close3(preds, labels):
    dataset_len = len(labels)


    ad_auc_array = []
    acc_array = []

    ad_1_array = []  #ad_sen
    ad_2_array = []  #ad_spec


    for i in range(2000):
        sample_index = []
        for j in range(2500):
            tmp_index = random.randint(0, dataset_len - 1)
            sample_index.append(tmp_index)
        test_y_ci = labels[sample_index]
        pred_ci = preds[sample_index]

        ad_fpr, ad_tpr, thresholds = roc_curve(test_y_ci, pred_ci[:, 1])  
        roc_auc_ad = auc(ad_fpr, ad_tpr)  

        acc = accuracy_score(test_y_ci, np.argmax(pred_ci, axis=1))

        ad_auc_array.append(roc_auc_ad)
        acc_array.append(acc)

        ad_sen = sensitivity2(test_y_ci, pred_ci[:, 1])
        ad_spec = specificity2(test_y_ci, pred_ci[:, 1])

        ad_1_array.append(ad_sen)
        ad_2_array.append(ad_spec)

    ad_auc_array = np.array(ad_auc_array)
    acc_array = np.array(acc_array)

    ad_1_array = np.array(ad_1_array)
    ad_2_array = np.array(ad_2_array)

    print("==========================AD===========================")
    print('auc:  ', np.percentile(ad_auc_array, 2.5), ',', np.percentile(ad_auc_array, 50), ',',
          np.percentile(ad_auc_array, 97.5))
    print('sen:  ', np.percentile(ad_1_array, 2.5), ',', np.percentile(ad_1_array, 50), ',',
          np.percentile(ad_1_array, 97.5))
    print('spec:  ', np.percentile(ad_2_array, 2.5), ',', np.percentile(ad_2_array, 50), ',',
          np.percentile(ad_2_array, 97.5))

    print('acc:  ', np.percentile(acc_array, 2.5), ',', np.percentile(acc_array, 50), ',',
          np.percentile(acc_array, 97.5))

    test_y_ad = labels
    ad_pred = preds

    fpr_ad, tpr_ad, thresholds = roc_curve(test_y_ad, ad_pred[:, 1])
    roc_auc_ad = auc(fpr_ad, tpr_ad)  # 求auc面积

    acc = accuracy_score(test_y_ad, np.argmax(ad_pred, axis=1))

    sen_ad = sensitivity2(test_y_ad, ad_pred[:, 1])
    spec_ad = specificity2(test_y_ad, ad_pred[:, 1])

    print("ACC= ", acc)
    print("%s sensitivity= %f specificity= %f auc= %f" % ('AD', sen_ad, spec_ad, roc_auc_ad))
    # print(acc,[sen, spec])

    font1 = {'family': 'Nimbus Roman',
             'weight': 'bold',
             'style': 'normal',
             'size': 15,
             }

    font2 = {'family': 'Nimbus Roman',
             'weight': 'bold',
             'style': 'normal',
             'size': 10,
             }


    plt.scatter([spec_ad], [spec_ad], s=50, label='AD operating point (Sensitivity={0:.4f})'.format(sen_ad))  # cn



    plt.plot(fpr_ad, tpr_ad, linewidth=1, color="blue",
             label='AD ROC (AUC = {0:.4f})'.format(roc_auc_ad))  

    plt.xlim([-0.05, 1.05])  
    plt.ylim([-0.05, 1.05])
    plt.xlabel('1-Specificity', font1)
    plt.ylabel('Sensitivity', font1)  
    plt.title('Dynamic-image-VGG', fontsize=15, fontweight='bold')
    plt.legend(loc="lower right", prop=font2)
    plt.savefig('Dynamic-image-VGG_close' + '_roc.jpg')
    plt.close()

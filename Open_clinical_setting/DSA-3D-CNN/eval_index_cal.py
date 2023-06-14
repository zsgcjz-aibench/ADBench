import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc, accuracy_score
import random
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt


#计算某个元素在列表中出现的次数
def countX(lst, x):
    count = 0
    for label in lst:
        if label == x:
            count += 1
    return count


#统计被预测正确的数量：preds.shape= (n, 2), labels.shape= (n, 1)
def correct_pred_count(preds, labels):
    labels = labels
    preds = preds
    un_pcount = 0
    ad_pcount = 0
    cn_pcount = 0
    for index, pred in enumerate(preds):
        if np.amax(pred) >= 0.95:   # 阈值 >= 0.95，才可以被诊断为AD和CN
            if np.amax(pred) == pred[0]: # 预测为cn
                if labels[index] == 0:  # 预测为cn, 真实为 cn
                    cn_pcount += 1
            if np.amax(pred) == pred[1]:  # 预测为ad
                if labels[index] == 1:  # 预测为ad, 真实为 ad
                    ad_pcount += 1
        else: # 预测为un
            if labels[index] == 2:  # 预测为un, 真实为 unknown
                un_pcount += 1

    print("cn_pcount= ", cn_pcount, "ad_pcount= ", ad_pcount, "un_pcount= ", un_pcount)

    return cn_pcount, ad_pcount, un_pcount


#准确率：所有test中被正确分类的概率：preds.shape= (n, 2), labels.shape= (n, 1)
def cal_acc(preds, labels):
    cn_pcount, ad_pcount, un_pcount = correct_pred_count(preds, labels)

    accuracy = (cn_pcount + ad_pcount + un_pcount) / len(labels)

    return accuracy


#敏感度：test中cn被正确分类的概率：preds.shape= (n, 2), labels.shape= (n, 1)
def cal_sensitivity(preds, labels):
    cn_pcount, ad_pcount, un_pcount = correct_pred_count(preds, labels)

    print("cn总数= ", countX(labels, 0))
    print("ad总数= ", countX(labels, 1))
    print("un总数= ", countX(labels, 2))


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

#特异度
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



#计算置信区间：preds= [0.6,0.9,0.76,0,95,...,0.99], labels=[0,1,0,1,...,1]
def cal_roc_curve(pred, test_y):
    fpr, tpr, thresholds = roc_curve(test_y, pred)  # 该函数得到伪正例、真正例、阈值，这里只使用前两个
    roc_auc = auc(fpr, tpr)  # 求auc面积

    return roc_auc


#one_hot编码--开放场景
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

#one_hot编码--封闭场景
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


#计算置信区间
def cal_CI(preds, labels):



    cn_auc_array = []
    ad_auc_array = []

    acc_array = []

    cn_sen_array = []
    ad_sen_array = []
    un_sen_array = []

    dataset_len = len(labels)
    print("dataset_len= ", dataset_len)
    total_pred = preds
    total_test_y = labels

    for i in range(2000):  # 进行2000次抽样
        sample_index = []
        for j in range(2500):  # 随机抽取2500个数据
            tmp_index = random.randint(0, dataset_len - 1)
            sample_index.append(tmp_index)
        test_y = total_test_y[sample_index]
        pred = total_pred[sample_index]

        ##label分别对cn, ad, unknow进行one_hot编码
        cn_label, ad_label, un_label = ont_hot_code(test_y)
        #计算auc
        # cn_preds = preds[0]
        # ad_preds = preds[1]
        # cn_auc = cal_roc_curve(pred[:, 0], cn_label)
        cn_fpr, cn_tpr, thresholds = roc_curve(pred[:, 0], cn_label)  # 该函数得到伪正例、真正例、阈值，这里只使用前两个
        cn_auc = auc(cn_fpr, cn_tpr)  # 求auc面积

        # ad_auc = cal_roc_curve(pred[:, 1], ad_label)
        ad_fpr, ad_tpr, thresholds = roc_curve(pred[:, 1], ad_label)  # 该函数得到伪正例、真正例、阈值，这里只使用前两个
        ad_auc = auc(ad_fpr, ad_tpr)  # 求auc面积
        print("cn_auc= ", cn_auc)
        print("ad_auc= ", ad_auc)


        #计算acc
        accuracy = cal_acc(pred, test_y)

        #计算sen
        cn_sen, ad_sen, un_sen = cal_sensitivity(pred, test_y)

        cn_auc_array.append(cn_auc)
        ad_auc_array.append(ad_auc)

        cn_sen_array.append(cn_sen)
        ad_sen_array.append(ad_sen)
        un_sen_array.append(un_sen)

        acc_array.append(accuracy)



    acc_array = np.array(acc_array)

    cn_auc_array = np.array(cn_auc_array)
    ad_auc_array = np.array(ad_auc_array)

    cn_sen_array = np.array(cn_sen_array)
    ad_sen_array = np.array(ad_sen_array)
    un_sen_array = np.array(un_sen_array)



    print('acc:  ', np.percentile(acc_array, 2.5), ',', np.percentile(acc_array, 50), ',',
          np.percentile(acc_array, 97.5))

    print('cn sen:  ', np.percentile(cn_sen_array, 2.5), ',', np.percentile(cn_sen_array, 50), ',',
          np.percentile(cn_sen_array, 97.5))
    print('ad sen:  ', np.percentile(ad_sen_array, 2.5), ',', np.percentile(ad_sen_array, 50), ',',
          np.percentile(ad_sen_array, 97.5))
    print('un sen:  ', np.percentile(un_sen_array, 2.5), ',', np.percentile(un_sen_array, 50), ',',
          np.percentile(un_sen_array, 97.5))

    print('cn auc:  ', np.percentile(cn_auc_array, 2.5), ',', np.percentile(cn_auc_array, 50), ',',
          np.percentile(cn_auc_array, 97.5))
    print('ad auc:  ', np.percentile(ad_auc_array, 2.5), ',', np.percentile(ad_auc_array, 50), ',',
          np.percentile(ad_auc_array, 97.5))


#开放场景
def get_score_final(ys_true, openmax_rate=[],
                    thr=0.7):  # ys_true=[test_y, test_y_cn, test_y_unkown], openmax_rate=[ad_pred, cn_pred , un_pred]三维数组（n, 3)
    nb_outputs = len(ys_true)  # 输出个数=3
    ys_pred_open_max = []

    if len(openmax_rate) > 0:  # = 3
        for i in range(nb_outputs):
            ys_pred_open_max.append([])  # ys_pred_open_max= [[], [] , []]

    for j in range(len(openmax_rate)):  # len(openmax_rate) = 3
        tmp_open_max = openmax_rate[j]  # openmax_rate=[ad_pred, cn_pred , un_pred]三维数组（n, 3)
        index = np.argmax(tmp_open_max)  # 计算每一列的最大索引下标,单列则原样返回

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
                
            if true_value == 0 and pred_value == 0:
                specificity_c += 1
                
            if (i == 0 or i == 1) and true_value == 1 and pred_value == 0 and ys_pred_open_max[2][k] == 1:
                f_unkown += 1

            if true_value == 0 and pred_value == 1:
                fpr += 1

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



#封闭场景
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

            # if (i==0 or i==1) and true_value == 1 and pred_value == 0 and ys_pred_open_max[2][k]==1:
            #    f_unkown+=1

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

#开放场景的ci计算函数
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

        ad_fpr, ad_tpr, thresholds = roc_curve(ad_test_y_ci, ad_pred_ci)  # AD的auc
        roc_auc = auc(ad_fpr, ad_tpr)  # 求auc面积
        fpr_cn, tpr_cn, thresholds_cn = roc_curve(cn_label_ci, cn_pred_ci)  # CN的auc
        roc_auc_cn = auc(fpr_cn, tpr_cn)  # 求auc面积
        #get_one_hot_2v([pred, 1 - pred, np.zeros(pred.size)], pred.size) 返回一个one-hot编码的[ad_pred, cn_pred , un_pred]三维数组（n, 3)
        acc, result = get_score_final([ad_test_y_ci, cn_label_ci, un_label_ci],get_one_hot_2v([ad_pred_ci, cn_pred_ci, np.zeros(len(cn_pred_ci))], len(cn_pred_ci)) )#np.zeros(pred.size)返回来一个pred.size形状和类型的用0填充的数组

        ad_auc_array.append(roc_auc)
        cn_auc_array.append(roc_auc_cn)
        acc_array.append(acc)

        ad_1_array.append(result[0][0]) #sen
        ad_2_array.append(result[0][1]) #spec
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


    fpr, tpr, thresholds = roc_curve(ad_test_y, ad_pred)    # 该函数得到伪正例、真正例、阈值，这里只使用前两个

    #np.save('/data/huangyunyou/result/closed_MRI_fpr_ad.npy', fpr)
    #np.save('/data/huangyunyou/result/closed_MRI_tpr_ad.npy', tpr)

    roc_auc = auc(fpr, tpr)  # 求auc面积

    fpr_cn, tpr_cn, thresholds_cn = roc_curve(test_y_cn, cn_pred)  # 该函数得到伪正例、真正例、阈值，这里只使用前两个

    #np.save('/data/huangyunyou/result/closed_MRI_fpr_cn.npy', fpr_cn)
    #np.save('/data/huangyunyou/result/closed_MRI_tpr_cn.npy', tpr_cn)

    roc_auc_cn = auc(fpr_cn, tpr_cn)  # 求auc面积
    #print(thresholds)
    #print(thresholds_cn)

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

    #, Unkown sen={1:.4f}, Global acc={2:.4f
    #, Unkown sen={1:.4f}, Global acc={2:.4f}
    #,result[2][0],acc
    plt.scatter([result[0][1]], [result[0][0]], s=50, label='AD operating point (Sensitivity={0:.4f})'.format(result[0][0])) #ad
    plt.scatter([result[1][1]], [result[1][0]], s=50, label='CN operating point (Sensitivity={0:.4f})'.format(result[1][0])) #cn
    plt.scatter([result[2][1]], [result[2][0]], s=50, label='Unkown operating point (Sensitivity={0:.4f})'.format(result[2][0]))  # cn

    #, markerfacecolor='red', markeredgecolor='gray'
    #, markerfacecolor='red', markeredgecolor='gray'
    plt.plot(fpr, tpr, linewidth=1, color="red", label='AD ROC (AUC = {0:.4f})'.format(roc_auc))    # 画出当前分割数据的ROC曲线
    plt.plot(fpr_cn, tpr_cn, linewidth=1, color="black", label='CN ROC (AUC = {0:.4f})'.format(roc_auc_cn))  # 画出当前分割数据的ROC曲线

    plt.xlim([-0.05, 1.05])  # 设置x、y轴的上下限，设置宽一点，以免和边缘重合，可以更好的观察图像的整体
    plt.ylim([-0.05, 1.05])
    plt.xlabel('1-Specificity',font1)
    plt.ylabel('Sensitivity',font1)  # 可以使用中文，但需要导入一些库即字体
    plt.title('Diagnosis with DSA-3D-CNN', fontsize=15,fontweight='bold')
    plt.legend(loc="lower right",prop=font2)
    plt.savefig('DSA-3D-CNN_open_roc.jpg')

#Fig s3 c 这个图和Diagnosis with baseline AI system中的图很像

#封闭场景下的CI计算函数
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

        ad_fpr, ad_tpr, thresholds = roc_curve(ad_test_y_ci, ad_pred_ci)  # AD的auc
        roc_auc = auc(ad_fpr, ad_tpr)  # 求auc面积
        fpr_cn, tpr_cn, thresholds_cn = roc_curve(cn_label_ci, cn_pred_ci)  # CN的auc
        roc_auc_cn = auc(fpr_cn, tpr_cn)  # 求auc面积
        #get_one_hot_2v([pred, 1 - pred, np.zeros(pred.size)], pred.size) 返回一个one-hot编码的[ad_pred, cn_pred , un_pred]三维数组（n, 3)
        acc, result = get_score_final_2v([ad_test_y_ci, cn_label_ci],get_one_hot_2v([ad_pred_ci, cn_pred_ci], len(cn_pred_ci)) )#np.zeros(pred.size)返回来一个pred.size形状和类型的用0填充的数组

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


    fpr, tpr, thresholds = roc_curve(ad_test_y, ad_pred)    # 该函数得到伪正例、真正例、阈值，这里只使用前两个
    
    df1 = pd.DataFrame(ad_test_y)
    df2 = pd.DataFrame(ad_pred)
    df1.to_csv('/home/lxs/DSA-3D-CNN/DSA-3D-CNN/DSA_ad_label.csv', index=0)
    df2.to_csv('/home/lxs/DSA-3D-CNN/DSA-3D-CNN/DSA_ad_pred.csv', index=0)
    # np.save('/data/huangyunyou/result/DSA_ad_label.npy', df1)
    # np.save('/data/huangyunyou/result/DSA_ad_pred.npy', df2)

    roc_auc = auc(fpr, tpr)  # 求auc面积

    
    

    fpr_cn, tpr_cn, thresholds_cn = roc_curve(test_y_cn, cn_pred)  # 该函数得到伪正例、真正例、阈值，这里只使用前两个

    #np.save('/data/huangyunyou/result/closed_MRI_fpr_cn.npy', fpr_cn)
    #np.save('/data/huangyunyou/result/closed_MRI_tpr_cn.npy', tpr_cn)

    roc_auc_cn = auc(fpr_cn, tpr_cn)  # 求auc面积
    #print(thresholds)
    #print(thresholds_cn)

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
    plt.plot(fpr, tpr, linewidth=1, color="black", label='DSA-3D-CNN(AUC = {0:.4f})'.format(roc_auc),
             linestyle='--')  # 画出当前分割数据的ROC曲线

    plt.xlim([-0.05, 1.05])  # 设置x、y轴的上下限，设置宽一点，以免和边缘重合，可以更好的观察图像的整体
    plt.ylim([-0.05, 1.05])
    plt.xlabel('1-Specificity',font1)
    plt.ylabel('Sensitivity',font1)  # 可以使用中文，但需要导入一些库即字体
    plt.title('Diagnosis with DSA-3D-CNN', fontsize=15,fontweight='bold')
    plt.legend(loc="lower right",prop=font2)
    plt.savefig('DSA-3D-CNN_close_roc.jpg')

#纯close，对于CN, AD
def cal_CI_plot_close3(preds, labels):
    dataset_len = len(labels)

    # print("labels= ", labels)
    #
    # cn_label, ad_label = ont_hot_code_close(labels)
    # labels = np.concatenate((cn_label, ad_label), axis=1)
    # labels = to_categorical(labels)
    # print("labels.shape= ", np.shape(labels))
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

        ad_fpr, ad_tpr, thresholds = roc_curve(test_y_ci, pred_ci[:, 1])  # AD的auc
        roc_auc_ad = auc(ad_fpr, ad_tpr)  # 求auc面积

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
             label='AD ROC (AUC = {0:.4f})'.format(roc_auc_ad))  # 画出当前分割数据的ROC曲线

    plt.xlim([-0.05, 1.05])  # 设置x、y轴的上下限，设置宽一点，以免和边缘重合，可以更好的观察图像的整体
    plt.ylim([-0.05, 1.05])
    plt.xlabel('1-Specificity', font1)
    plt.ylabel('Sensitivity', font1)  # 可以使用中文，但需要导入一些库即字体
    plt.title('Dynamic-image-VGG', fontsize=15, fontweight='bold')
    plt.legend(loc="lower right", prop=font2)
    plt.savefig('Dynamic-image-VGG_close' + '_roc.jpg')
    plt.close()

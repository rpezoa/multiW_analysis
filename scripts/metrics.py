from math import sqrt, log
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score, accuracy_score, roc_curve, precision_recall_curve,roc_auc_score
from sklearn.metrics import classification_report
import numpy as np
from matplotlib import pyplot as plt

def significance(s,b):
    ##Z[s,b]=[Sqrt[2 ((s+b) Log[1+s/b]-s)]]
    s = sqrt(2 * ((s + b) * log(1 + (s/b)) - s ))
    return s

def significance_with_weight(y_pred,weights):
    ##Z[s,b]=[Sqrt[2 ((s+b) Log[1+s/b]-s)]]
    mask_signal = y_pred == 1
    mask_background = y_pred == 0
    sig = y_pred[mask_signal]*weights[mask_signal]
    back = (1 - y_pred[mask_background])*weights[mask_background]
    s = sig.sum()
    #print("sig.sum()", sig.sum(), sig.shape)
    b= back.sum()
    #print("back.sum()", back.sum(), back.shape)
    signif = sqrt(2 * ((s + b) * log(1 + (s/b)) - s ))
    return signif

def significance_true_values(y_pred, y_true, weights):

    # TRUE POSITIVE and FALSE POSITIVE
    tp_mask = (y_pred == 1)   & (y_true == 1)

    fp_mask = (y_pred == 0) &  (y_true == 0 )

    tp = y_pred[tp_mask]
    fp = y_pred[fp_mask]

    weights_tp = weights[tp_mask]
    weights_fp = weights[fp_mask]


    #print(tp.shape, weights_tp.shape, fp.shape, weights_fp.shape)
    s = weights_tp.sum()
    b = weights_fp.sum()
    print("::: TP signal",s)
    print("::::::::::b", b)

    signif = sqrt(2 * ((s + b) * log(1 + (s/b)) - s ))
    return signif




def pred_with_threshold(y_pred, weights, y_true):
    big_sig = -100
    for t in np.linspace(0.05,0.95,11):
        y_new = y_pred >= t   
        #s = y_new.sum()
        #b = y_new.shape[0] - s
        
        #sig = significance(s,b)
        #print("signal: ", s, "background:", b, "S:::::", sig)
        #sig = significance_with_weight(y_new,weights)
        sig = significance_true_values(y_new, y_true, weights)
        print("Significance :::::",sig )
        
        if sig >= big_sig:
            print("yes",sig)
            big_sig = sig
            the_t = t
    print("Best threshold:::", the_t, "with SIGNIFICANCE=", big_sig)
    return the_t


def prediction(y_test, y_pred_score, title,weights):

    #print(y_test.shape, y_pred.shape, y_pred_score)


    th = pred_with_threshold(y_pred_score, weights, y_test)
    print("Predicting with threshold:", th)

    y_pred = y_pred_score >= th
    y_pred = y_pred.astype(int)


    f1 = round(f1_score(y_test, y_pred),2)
    prec = round(precision_score(y_test, y_pred),2)
    rec = round(recall_score(y_test, y_pred),2)
    acc = round(accuracy_score(y_test, y_pred),2)
    cm = confusion_matrix(y_test, y_pred)
    s = y_pred.sum()
    b = y_pred.shape[0] - s
    print("signal: ", s, "background:", b)

    sig_w = significance_with_weight(y_pred,weights)
    print("SIGNIFICANCE WITH WEIGHTS", sig_w)

    fpr, tpr, thresholds = roc_curve(y_test, y_pred_score)

    plt.figure()
    plt.title("ROC curve, "+title)
    plt.plot(fpr, tpr, '-b')
    plt.xlabel("fpr")
    plt.ylabel("tpr")
    plt.show()

    roc_auc= roc_auc_score(y_test, y_pred_score)

    p,r,t = precision_recall_curve(y_test, y_pred_score)

    plt.figure()
    plt.title("PR curve, "+title)
    plt.plot(r, p, '-b')
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.show()

    print("f1:",f1, "prec:",prec,"rec:", rec, "acc:",acc, "roc_auc",roc_auc)
    print(cm)
    return {"f1":f1, "prec":prec,"rec": rec, "acc": acc, "cm":cm, "roc_auc":roc_auc,"sig":sig_w,"threshold":th}






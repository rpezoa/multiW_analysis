from metrics import significance, significance_with_weight, significance_true_values
import pandas as pd
import numpy as np

def fromcsv2df(signal_file):
    df = pd.read_csv(signal_file)
    return df


def distribution(dict_labels):
    # data distribution per class
    print("Distribution of classes:")
    l=[]
    for key in dict_labels:
        unique, counts = np.unique(dict_labels[key], return_counts=True)
        l.append({"data":key.split("_")[-1],"background":counts[0], "signal":counts[1],
                  "% b":counts[0]/counts.sum(), "% s":counts[1]/counts.sum(),  "total": counts[0] + counts[1]})
    df_distro = pd.DataFrame(l)
    return df_distro

def distribution_weights(dict_labels, dict_weights):
    l=[]
    for key in dict_labels:
        unique, counts = np.unique(dict_labels[key], return_counts=True)
        N_b = counts[0]
        N_s = counts[1]
        N_p_b = N_b /counts.sum() * 100
        N_p_s = N_s/counts.sum() * 100
        N_total = N_s + N_b
        
        mask_y_s = dict_labels[key]==1
        weight_s  = dict_weights[key][mask_y_s]

        mask_y_b = dict_labels[key] == 0
        weight_b = dict_weights[key][mask_y_b]
        
        N_w_b  = weight_b.sum()
        N_w_s = weight_s.sum()
        N_w_t = N_w_b + N_w_s

        signif_true = significance_true_values(dict_labels[key], dict_labels[key], dict_weights[key])


        l.append({"data":key.split("_")[-1],"#background":N_b, "#signal":N_s,
            "% b":N_p_b, "% s":N_p_s,  "total": N_total, "w_s": N_w_s, "w_b":N_w_b, "%w_s":(N_w_s/N_w_t)*100,
            "%w_b":(N_w_b/N_w_t)*100 ,"w_total": N_w_t, 
            "SIGNIF_WEIGHT": significance_with_weight(dict_labels[key],dict_weights[key]),
            "SIGNIF": significance(N_s,N_b), "SIGNIF_TRUE": signif_true})

    
    df_distro = pd.DataFrame(l)
    return df_distro

def filtering(df,X,y,weights, name_colum_to_filter):
    print("Category:", name_colum_to_filter)
    mask = df[name_colum_to_filter] == 1
    X = X[mask]
    y = y[mask]
    weights = weights[mask]
    return X,y, weights



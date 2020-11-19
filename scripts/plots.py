from matplotlib import pyplot as plt
import numpy as np

def plot_scores(y_pred_score, y_test, y_pred_score_train, y_train, show=False):
    mask_sig = y_test == 1
    mask_back = y_test == 0
    
    mask_sig_train = y_train == 1
    mask_back_train = y_train == 0
    
    hmin=y_pred_score.min()
    hmax=y_pred_score.max()

    if show:
        
        plt.figure(figsize=(10,5))

        ax1 = plt.subplot(1,2,1)# number of row = 1, number of columns = 2, subplot order = 1 
        # OR can be written as ax1 = plt.subplot(121)

        
        plt.figure()
        ax1.hist(y_pred_score[mask_sig], density=False, bins=40, range=(hmin,hmax), label="Signal test")
        ax1.hist(y_pred_score[mask_back], density=False, bins=40, label="Background test")
        ax1.legend(loc="best")
        plt.show()

        
        ax2 = plt.subplot(1,2,2)
        ax2.hist(y_pred_score_train[mask_sig_train], density=False, bins=40, label="Signal train", alpha=0.5)
        ax2.hist(y_pred_score_train[mask_back_train], density=False, bins=40, label="Background train", alpha=0.5)
        ax2.legend(loc="best")
        
        plt.grid()
        plt.show()
    return mask_sig, mask_back, mask_sig_train, mask_back_train



def plot_scores_v2(y_pred_score, mask_sig,mask_back,y_pred_score_train, mask_sig_train, mask_back_train,met_test, met_train, category,signal):
#https://dbaumgartel.wordpress.com/2014/03/14/machine-learning-examples-scikit-learn-versus-tmva-cern-root/



    plt.figure()
    tn = met_test["cm"][1,0]
    fp = met_test["cm"][0,1]
    #ax1 = plt.subplot(111)
    number_of_bins = 15
    # Draw solid histograms for the training data
    n,bins, patches = plt.hist(y_pred_score[mask_sig], density=False, bins=number_of_bins,facecolor='blue', label="Signal")

    n2,bins2, patches2 = plt.hist(y_pred_score[mask_back], density=False, bins=number_of_bins, facecolor='red', label="Background")
    print(n2,"\n", n)
    plt.title(signal +"_" + category + "_Testing, rec: " + str(np.round(met_test["rec"],2)) + " prec: " + str(np.round(met_test["prec"],2)) + " f1: " + str(np.round(met_test["f1"],2)) + " SIG: " + str(np.round(met_test["sig"],2)) + " FP: " + str(fp) + " FN: " + str(tn))
    plt.xlabel("Classification score RF")
    plt.ylabel("Counts/Bin")
    plt.legend(loc='upper center', shadow=True,ncol=2)
    plt.savefig("../output/"+ signal +"_"+category + "_testing.png" )
    plt.show()

    plt.figure()
    plt.hist(y_pred_score_train[mask_sig_train], bins=number_of_bins,density=False,facecolor='blue', label="Signal train")
    n3,bins3, patches3 = plt.hist(y_pred_score_train[mask_back_train], bins=number_of_bins, density=False,facecolor='red',label="Background train")
    tn = met_train["cm"][1,0]
    fp = met_train["cm"][0,1]
    plt.title(signal +"_" + category + ", Train, rec: " + str(np.round(met_train["rec"],2)) + " prec: " + str(np.round(met_train["prec"],2)) + " f1: " + str(np.round(met_train["f1"],2)) + " SIG: " + str(np.round(met_train["sig"],2)) + " FP: " + str(fp) + " FN: " + str(tn))
    plt.xlabel("Classification score RF")
    plt.ylabel("Counts/Bin")
    plt.legend(loc='upper center', shadow=True,ncol=2)
    plt.savefig("../output/"+ signal +"_"+ category + "_training.png" )
    plt.show()

    bin_centers = ( bins[:-1] + bins[1:]  ) /2.
    bin_widths = (bins[1:] - bins[:-1])

    ErrorBar_testing_S = np.sqrt(n)
    ErrorBar_testing_B = np.sqrt(n2)

    #ax1.errorbar(bin_centers, n, yerr=ErrorBar_testing_S, xerr=None, ecolor='cyan',c='cyan',fmt='o',label='S (Test)')
    #ax1.errorbar(bin_centers, n2, yerr=ErrorBar_testing_B, xerr=None, ecolor='magenta',c='magenta',fmt='o',label='B (Test)')


    # Make labels and title
    #plt.title("Classification")
    #plt.xlabel("Classification score RF")
    #plt.ylabel("Counts/Bin")

    #legend = ax1.legend(loc='upper center', shadow=True,ncol=2)
    #for alabel in legend.get_texts():
    #            alabel.set_fontsize('small')

    c_min = bins.min()
    AllHistos = [bins,bins2,bins3]
    c_max = max([histo.max() for histo in AllHistos])*1.05

    h_min = n3.min()
    AllHistos = [n,n2,n3]
    h_max = max([histo.max() for histo in AllHistos])*1.2

    #ax1.axis([c_min, c_max, h_min, h_max])

    #ax1.axvspan(0.05, 0.35, color='blue',alpha=0.08)
    #ax1.axvspan(0.05,0.05, color='red',alpha=0.08)

    #plt.show()



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve, auc


def readCSV(fileName):
    return pd.read_csv(fileName, header=None, delimiter=",")


def drawPR():
    # NRLLDAI
    NRLLabel = []
    NRLScore = []

    probFilePath = []
    for i in range(10):
        filePath = "../FinalMode/Result/_XGB_第%d次Prob.csv"%(i+1)
        probFilePath.append(filePath)
    for path in probFilePath:
        data = pd.read_csv(path, header=None)
        for row in data.itertuples():
            NRLLabel.append(getattr(row, "_1"))
            NRLScore.append(getattr(row, "_2"))

    label = np.array(NRLLabel)
    prob = np.array(NRLScore)
    pre, rec, _ = precision_recall_curve(label, prob)
    AP = average_precision_score(label, prob, average='macro', pos_label=1, sample_weight=None)
    plt.plot(pre, rec, color="red",
             label='NRLLDAI (AUPR = %0.4f)' % (AP),
             lw=2, alpha=1)

    # LDASR
    label = np.array(np.load('LDASR-label.npy'))
    prob = np.array(np.load("LDASR-pred.npy"))
    pre, rec, _ = precision_recall_curve(label, prob)
    AP = average_precision_score(label, prob, average='macro', pos_label=1, sample_weight=None)
    plt.plot(pre, rec, color="yellow",
             label=r'LDASR (AUPR = %0.4f)' % (AP),
             lw=2, alpha=1)

    # NCPHLDA
    label = np.array(readCSV("NCPHLDA_label.txt"))
    prob = np.array(readCSV("NCPHLDA_predict.txt"))
    pre, rec, _ = precision_recall_curve(label, prob)
    AP = average_precision_score(label, prob, average='macro', pos_label=1, sample_weight=None)
    plt.plot(pre, rec, color="GREEN",
             label=r'NCPHLDA (AUPR = %0.4f)' % (AP),
             lw=2, alpha=1)

    # SDLDA
    label = np.array(np.load('SDLDAlabel.npy'))
    prob = np.array(np.load("SDLDApred.npy"))
    pre, rec, _ = precision_recall_curve(label, prob, pos_label=1)
    AP = average_precision_score(label, prob, average='macro', pos_label=1, sample_weight=None)
    plt.plot(pre, rec, color="blue",
             label=r'SDLDA (AUPR = %0.4f)' % (AP),
             lw=2, alpha=1)

    # TPGLDA
    label = np.array(readCSV("TPGLDA_label.csv"))
    prob = np.array(readCSV("TPGLDA_predict.csv"))
    pre, rec, _ = precision_recall_curve(label, prob)
    AP = average_precision_score(label, prob, average='macro', pos_label=1, sample_weight=None)
    plt.plot(pre, rec, color="deeppink",
             label=r'TPGLDA (AUPR = %0.4f)' % (AP),
             lw=2, alpha=1)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0, 1])
    plt.xlim([0, 1])
    plt.legend(loc='lower right')
    plt.savefig('PRCompare.svg')
    plt.savefig('PRCompare.tif')
    plt.show()


def drawAUC():
    pass
    # NRLLDAI
    NRLLabel = []
    NRLScore = []

    probFilePath = []
    for i in range(10):
        filePath = "../FinalMode/Result/_XGB_第%d次Prob.csv" % (i + 1)
        probFilePath.append(filePath)
    for path in probFilePath:
        data = pd.read_csv(path, header=None)
        for row in data.itertuples():
            NRLLabel.append(getattr(row, "_1"))
            NRLScore.append(getattr(row, "_2"))

    label = np.array(NRLLabel)
    prob = np.array(NRLScore)
    fpr, tpr, _ = roc_curve(label, prob)
    AUC = auc(fpr, tpr)
    plt.plot(fpr, tpr, color="red",
             label='NRLLDAI (AUC = %0.4f)' % AUC,
             lw=2, alpha=1)

    # LDASR
    label = np.array(np.load('LDASR-label.npy'))
    prob = np.array(np.load("LDASR-pred.npy"))
    fpr, tpr, _ = roc_curve(label, prob)
    AUC = auc(fpr, tpr)
    plt.plot(fpr, tpr, color="yellow",
             label=r'LDASR (AUPR = %0.4f)' % (AUC),
             lw=2, alpha=1)

    # NCPHLDA
    label = np.array(readCSV("NCPHLDA_label.txt"))
    prob = np.array(readCSV("NCPHLDA_predict.txt"))
    fpr, tpr, _ = roc_curve(label, prob)
    AUC = auc(fpr, tpr)
    plt.plot(fpr, tpr, color="GREEN",
             label=r'NCPHLDA (AUPR = %0.4f)' % (AUC),
             lw=2, alpha=1)

    # SDLDA
    label = np.array(np.load('SDLDAlabel.npy'))
    prob = np.array(np.load("SDLDApred.npy"))
    fpr, tpr, _ = roc_curve(label, prob)
    AUC = auc(fpr, tpr)
    plt.plot(fpr, tpr, color="blue",
             label=r'SDLDA (AUPR = %0.4f)' % (AUC),
             lw=2, alpha=1)

    # TPGLDA
    label = np.array(readCSV("TPGLDA_label.csv"))
    prob = np.array(readCSV("TPGLDA_predict.csv"))
    fpr, tpr, _ = roc_curve(label, prob)
    AUC = auc(fpr, tpr)
    plt.plot(fpr, tpr, color="deeppink",
             label=r'TPGLDA (AUPR = %0.4f)' % (AUC),
             lw=2, alpha=1)


    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.savefig("AUCCompare.tiff")
    # plt.savefig("modelcompare.pdf")
    plt.savefig("AUCCompare.png")
    plt.show()


if __name__ == "__main__":
    drawPR()
    #drawAUC()

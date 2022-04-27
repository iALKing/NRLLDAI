import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve, auc



#方法名：DeepWalk、GF、GRAREP、Hope、LAP、Line、LLE、Node2vec【Node2vec】、SDNE

def getPath(method, times):
        filePath = "./Result/_%s_第%d次Prob.csv" % (method, times)
        return filePath


def getList(method):
    label = []
    score= []
    for i in range(10):
        resultPath = getPath(method, i+1)
        readFile = pd.read_csv(resultPath, header=None)
        for row in readFile.itertuples():
            label.append(getattr(row, "_1"))
            score.append(getattr(row, "_2"))
    return label, score

def drawPR(methodList, colorList):
    for i in range(len(methodList)+1):
        if i == len(methodList):
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.ylim([0, 1])
            plt.xlim([0, 1])
            plt.legend(loc='lower right')
            plt.savefig('ModelCompare.svg')
            plt.savefig('ModelCompare.tif')
            plt.show()
        else:
            label, prob = getList(methodList[i])
            label = np.array(label)
            prob = np.array(prob)
            pre, rec, _ = precision_recall_curve(label, prob)
            AP = average_precision_score(label, prob, average='macro', pos_label=1, sample_weight=None)
            plt.plot(pre, rec, color=colorList[i],
             label='%s (AUPR = %0.4f)' % (methodList[i], AP),
             lw=2, alpha=1)


def drawROC(methodList, colorList):
    for i in range(len(methodList)+1):
        if i == len(methodList):
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.legend(loc='lower right')
            plt.savefig("modelcompare.tiff")
            # plt.savefig("modelcompare.pdf")
            plt.savefig("modelcompare.png")
            plt.show()
        else:
            label, prob = getList(methodList[i])
            label = np.array(label)
            prob = np.array(prob)
            fpr, tpr, _ = roc_curve(label, prob)
            AUC = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=colorList[i],
             label='%s (AUC = %0.4f)' % (methodList[i], AUC),
             lw=2, alpha=1)


if __name__=="__main__":
    # 分类器
    classList = ["LR", "NB", "RF", "ADB", "XGB"]
    colorList = ["red",  "yellow", "green", "blue",  "deeppink"]
    drawPR(classList, colorList)
    #drawROC(classList, colorList)



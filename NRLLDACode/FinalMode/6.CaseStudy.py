import numpy as np
#import pandas as pd
import random
import csv
import math
import xgboost as xgb
from sklearn.metrics import confusion_matrix


def ReadMyCsv(SaveList, fileName):
    csv_reader = csv.reader(open(fileName))
    for row in csv_reader:  # 把每个rna疾病对加入OriginalData，注意表头
        SaveList.append(row)
    return

def ReadMyCsv2(SaveList, fileName):
    csv_reader = csv.reader(open(fileName, encoding='utf-8'))
    for row in csv_reader:
        counter = int(0)
        while counter < len(row):
            row[counter] = int(row[counter])  # 转换数据类型
            counter = counter + 1
        SaveList.append(row)
    return


def ReadMyCsv3(SaveList, fileName):
    csv_reader = csv.reader(open(fileName, encoding='utf-8-sig'))
    for row in csv_reader:
        counter = 0
        while counter < len(row):
            row[counter] = float(row[counter])  # 转换数据类型
            counter = counter + 1
        SaveList.append(row)
    return


def StorFile(data, fileName):
    with open(fileName, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
    return


def geneLabel(Sample):
    Label = []
    counter = 0
    while counter < (len(Sample) / 2):
        Label.append(1)
        counter = counter + 1
    counter = 0
    while counter < (len(Sample) / 2):
        Label.append(0)
        counter = counter + 1
    return Label


def myConfuse(SampleFeature, SampleLabel):
    # 打乱数据集顺序

    counter = 0
    R = []
    while counter < len(SampleFeature):
        R.append(counter)
        counter = counter + 1
    random.shuffle(R)

    RSampleFeature = []
    RSampleLabel = []
    counter = 0
    while counter < len(R):
        RSampleFeature.append(SampleFeature[R[counter]])
        RSampleLabel.append(SampleLabel[R[counter]])
        counter = counter + 1
    return RSampleFeature, RSampleLabel

def myConfusionMatrix(y_real, y_predict):
    CM = confusion_matrix(y_real, y_predict).tolist()
    # print(CM)
    TN = CM[0][0]
    FP = CM[0][1]
    FN = CM[1][0]
    TP = CM[1][1]
    print('TN:%d, FP:%d, FN:%d, TP:%d' % (TN, FP, FN, TP))

    Acc = (TN + TP) / (TN + TP + FN + FP)
    Sen = TP / (TP + FN)
    Spec = TN / (TN + FP)
    Prec = TP / (TP + FP)
    MCC = (TP * TN - FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))

    # 分母可能出现0，需要讨论待续
    print('Acc:', round(Acc, 4))
    print('Sen:', round(Sen, 4))
    print('Spec:', round(Spec, 4))
    print('Prec:', round(Prec, 4))
    print('MCC:', round(MCC, 4))

    Result = []
    Result.append(round(Acc, 4))
    Result.append(round(Sen, 4))
    Result.append(round(Spec, 4))
    Result.append(round(Prec, 4))
    Result.append(round(MCC, 4))

    return Result


def myAverage(matrix):
    SumAcc = 0
    SumSen = 0
    SumSpec = 0
    SumPrec = 0
    SumMcc = 0

    counter = 0
    while counter < len(matrix):
        SumAcc = SumAcc + matrix[counter][0]
        SumSen = SumSen + matrix[counter][1]
        SumSpec = SumSpec + matrix[counter][2]
        SumPrec = SumPrec + matrix[counter][3]
        SumMcc = SumMcc + matrix[counter][4]
        counter = counter + 1

    print('AverageAcc:', SumAcc / len(matrix))
    print('AverageSen:', SumSen / len(matrix))
    print('AverageSpec:', SumSpec / len(matrix))
    print('AveragePrec:', SumPrec / len(matrix))
    print('AverageMcc:', SumMcc / len(matrix))

    return


def myRealAndPredictionProb(Real, prediction):
    RealAndPredictionProb = []

    counter = 0
    while counter < len(prediction):
        pair = []
        pair.append(Real[counter])
        pair.append(prediction[counter][1])
        RealAndPredictionProb.append(pair)
        counter = counter + 1

    return RealAndPredictionProb


def myRealAndPrediction(Real, prediction):
    RealAndPrediction = []

    counter = 0
    while counter < len(prediction):
        pair = []
        pair.append(Real[counter])
        pair.append(prediction[counter])
        RealAndPrediction.append(pair)
        counter = counter + 1

    return RealAndPrediction


def myStd(result):
    NewMatrix = []

    counter = 0
    while counter < len(result[0]):
        row = []
        NewMatrix.append(row)
        counter = counter + 1

    counter = 0
    while counter < len(result):
        counter1 = 0
        while counter1 < len(result[counter]):
            NewMatrix[counter1].append(result[counter][counter1])
            counter1 = counter1 + 1
        counter = counter + 1

    StdList = []
    MeanList = []
    counter = 0
    while counter < len(NewMatrix):
        # std
        arr_std = np.std(NewMatrix[counter], ddof=1)
        StdList.append(arr_std)
        # mean
        arr_mean = np.mean(NewMatrix[counter])
        MeanList.append(arr_mean)
        counter = counter + 1
    result.append(MeanList)
    result.append(StdList)

    # 换算成百分比制
    counter = 0
    while counter < len(result):
        counter1 = 0
        while counter1 < len(result[counter]):
            result[counter][counter1] = round(result[counter][counter1] * 100, 2)
            counter1 = counter1 + 1
        counter = counter + 1

    return result

if __name__ == '__main__':
    positiveSample = []
    ReadMyCsv2(positiveSample, 'PositiveSample.csv')
    negativeSample = []
    ReadMyCsv2(negativeSample, 'NeagtiveSample.csv')
    newRandomList = []
    ReadMyCsv2(newRandomList, 'RandomListGroup.csv')

    allFeature = []
    ReadMyCsv3(allFeature, '../Data/LMDNFeature/FinalFeature.csv')

    caseFeature = allFeature[255]
    # 49 BreastNeoplasms
    #95 ColorectalNeoplasms
    # 255 lungNeopla

    testFeature = []
    print(caseFeature)
    lncNum = 432


    while lncNum < len(allFeature):
        featurePair = []
        featurePair.extend(caseFeature)
        featurePair.extend(allFeature[lncNum])
        testFeature.append(featurePair)
        lncNum = lncNum + 1

    positiveSampleFeature = []
    counter1 = 0
    while counter1 < len(positiveSample):
        FeaturePair = []
        FeaturePair.extend(allFeature[positiveSample[counter1][0]])
        FeaturePair.extend(allFeature[positiveSample[counter1][1]])
        positiveSampleFeature.append(FeaturePair)
        counter1 = counter1 + 1

    negativeSampleFeature = []
    counter1 = 0
    while counter1 < len(negativeSample):
        FeaturePair = []
        FeaturePair.extend(allFeature[negativeSample[counter1][0]])
        FeaturePair.extend(allFeature[negativeSample[counter1][1]])
        negativeSampleFeature.append(FeaturePair)
        counter1 = counter1 + 1

    trainFeature = []
    trainFeature = positiveSampleFeature
    trainFeature.extend(negativeSampleFeature)
    trainLabel = geneLabel(trainFeature)
    trainFeature, trainLabel = myConfuse(trainFeature, trainLabel)

    model = xgb.XGBClassifier(use_label_encoder=False, max_depth=2, min_child_weight=50, subsample=0.3)
    trainFeature = np.array(trainFeature)
    trainLabel = np.array(trainLabel)
    model.fit(trainFeature, trainLabel)

    y_score0 = model.predict(np.array(testFeature))
    y_score1 = model.predict_proba(np.array(testFeature))

    np.savetxt('_CS_ LungNeopla.csv', y_score0)
    np.savetxt('_CS_ LungNeopla.csv', y_score1)








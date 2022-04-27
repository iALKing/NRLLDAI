import csv


def readNode(nodeList, fileName):
    csv_reader = csv.reader(open(fileName,encoding='utf-8-sig'))
    for row in csv_reader:
        nodeList.extend(row)
    return


def readEdge(edgeList, fileName):
    csv_reader = csv.reader(open(fileName,encoding='utf-8-sig'))
    for row in csv_reader:
        edgeList.append(row)
    return


#保存数据
def storeFile(data, fileName):
    with open(fileName, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
    return


def edgeNum(nodeFile, edgeFile):
    allNode = []
    allEdge = []
    readNode(allNode, nodeFile)     #print(allNode[0])
    readEdge(allEdge, edgeFile)     #print(allEdge[0][1])
    allEdgeNum = []

    for (node1, node2) in allEdge:
        numPair = []
        for node in allNode:
            if node == node1:
                numPair.append(allNode.index(node))
                break
        for node in allNode:
            if node == node2:
                numPair.append(allNode.index(node))
                break
        allEdgeNum.append(numPair)

    #print(allEdgeNum[0])
    storeFile(allEdgeNum, "AllEdgeNum.csv")


if __name__ =="__main__":
    nodeFile = "../Data/SamLDData/AllNode.csv"
    edgeFile = "../Data/SamLDData/AllEdge.csv"
    edgeNum(nodeFile, edgeFile)



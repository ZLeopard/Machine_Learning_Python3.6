from sklearn.feature_extraction import DictVectorizer
import csv
from sklearn import tree
from sklearn import preprocessing
import graphviz

from sklearn.externals.six import StringIO

# Read in the csv file and put features into list of dict and list of class label
allElectronicsData = open(r'./AllElectronics.csv', 'r')
reader = csv.reader(allElectronicsData)
print(type(reader))
headers = next(reader)   # Python3中改为next(iterator)

print(headers)

featureList = []      # 特征向量列表
labelList = []           # 标记列表

for row in reader:
    labelList.append(row[len(row)-1])
    rowDict = {}
    for i in range(1, len(row)-1):    # range函数左闭右开
        rowDict[headers[i]] = row[i]
    featureList.append(rowDict)

print(featureList)

# Vetorize features
vec = DictVectorizer()
dummyX = vec.fit_transform(featureList).toarray()

print("dummyX: " + str(dummyX))
print(vec.get_feature_names())

print("labelList: " + str(labelList))

# vectorize class labels
lb = preprocessing.LabelBinarizer()
dummyY = lb.fit_transform(labelList)
print("dummyY: " + str(dummyY))

# Using decision tree for classification
# clf = tree.DecisionTreeClassifier()
clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(dummyX, dummyY)
print("clf: " + str(clf))   # str可以打印函数的信息，要会用


# Visualize model

dot_data = tree.export_graphviz(clf, out_file=None, feature_names=vec.get_feature_names())  # 最后一个参数要还原原来变量名字
graph = graphviz.Source(dot_data)
graph.render("iris")

oneRowX = dummyX[0, :]
print("oneRowX: " + str(oneRowX))

newRowX = oneRowX
newRowX[0] = 1
newRowX[2] = 0     # 把第一行youth改为middle_age
print("newRowX: " + str(newRowX))

predictedY = clf.predict(newRowX.reshape(1 ,-1))      # array的reshape函数中的-1表示未知，reshape(1,-1)表示函数为一行，列数未知（Python 自动计算）
print("predictedY: " + str(predictedY))



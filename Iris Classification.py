import numpy as np
from numpy import genfromtxt
from sklearn import linear_model

file = genfromtxt('Iris.csv', delimiter = ',' , dtype='str')
file = file[1:]

dic = {}
count =0
for n in file:
    if n[5] not in dic:
        dic[n[5]] = count
        count +=1

for n in file:
    n[5] = dic[n[5]]
    
trainingSet = file[:130]
testingSet = file[130:]

trainingX = trainingSet[:,[1,2,3,4]]
trainingX = trainingX.astype(float)
#print(trainingX)
trainingY = trainingSet[:,[5]]
#print(trainingY)

testingX = testingSet[:,[1,2,3,4]]
testingX = testingX.astype(float)
testingY = testingSet[:,[5]]

model = linear_model.LogisticRegression()
model.fit(trainingX,trainingY)

print('predicted value is',model.predict([testingX[12]]))
print("Real Value is", str(testingY[12]))

accurecy = model.score(testingX, testingY) * 100
print(accurecy)

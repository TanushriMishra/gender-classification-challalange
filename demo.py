import numpy as np
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
clf = tree.DecisionTreeClassifier()
#3 classifier and defing objects of these models
clff = KNeighborsClassifier()
clfff = SVC()
clffff = GaussianNB()



# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']


# CHALLENGE - ...and train them on our data
clf = clf.fit(X, Y)

prediction = clf.predict([[190, 70, 43]])

clff = clff.fit(X, Y)
predictionn = clff.predict([[190, 70, 43]])

clfff = clfff.fit(X, Y)
predictionnn = clfff.predict([[190, 70, 43]])

clffff = clffff.fit(X, Y)
predictionnnn = clffff.predict([[190, 70, 43]])
# CHALLENGE compare their reusults and print the best one!
#print all of them
print(prediction)
print(predictionn)
print(predictionnn)
print(predictionnnn)
#Defing test set
_X=[[184,84,44],[198,92,48],[183,83,44],[166,47,36],[170,60,38],[172,64,39],[182,80,42],[180,80,43]]
_Y=['male','male','male','female','female','female','male','male']
#Predicting the class to which the particular data belongs
prediction = clf.predict(_X)
print(prediction)
acc1 = accuracy_score(_Y,prediction) * 100
print('Accuracy of Decision tree: {}%'.format(acc1))

predictionn = clff.predict(_X)
print(predictionn)
acc2 = accuracy_score(_Y,predictionn) * 100
print('KNeighborsClassifier: {}%'.format(acc2))

predictionnn = clfff.predict(_X)
print(predictionnn)
acc3 = accuracy_score(_Y,predictionnn) * 100
print('Accuracy of SVC: {}%'.format(acc3))

predictionnnn = clffff.predict(_X)
print(predictionnnn)
acc4 = accuracy_score(_Y,predictionnnn) * 100
print('Accuracy of GaussianNB: {}%'.format(acc4))

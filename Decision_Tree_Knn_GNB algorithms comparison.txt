

import pandas as pd

from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split 


#this command imports the data
dataset=pd.read_csv('skyserver.csv')

#on this step i drop the data that they are not usefull for the analysis as is the same number for all entries
dataset.drop(['objid'],axis = 1, inplace =True)

print ("Dataset: ")
#prints the first 5 rows
print(dataset.head())
#print the number of rows and columns 
print(dataset.shape)





#the class column on the data set is string so it can not work as expected so i convert them into string with codes 1.2.3
#this function is used to change the string to integer
def modify_to_integer(category):
    if category=='STAR':
        return 1
    elif category=='GALAXY':
        return 2
    else:
        return 3
    
#I used this code to assign a numerical code 
dataset['category'] = dataset['class'].apply(modify_to_integer)

#after i assigned a numerical code i will drop the data of class 
#so my code runs as expcted
dataset.drop(['class'],axis=1,inplace=True)


# Seperating the target variable 
X = dataset.drop('category', axis=1)
y = dataset['category']

# divide the data in train and test data.Test data set to 30% and train to 70%
X_train, X_test, y_train, y_test = train_test_split( 
	X, y, test_size = 0.3,random_state = 50 ) 



#I fit the data with gini classifier
giniclassifier = DecisionTreeClassifier(criterion = 'gini', random_state = 50 )
giniclassifier.fit(X_train, y_train)

#I fit the data with entropy classifier
entropyclassifier= DecisionTreeClassifier(criterion = 'entropy', random_state = 50)
entropyclassifier.fit(X_train, y_train) 


from sklearn.neighbors import KNeighborsClassifier
#I fit the data with KNN classifier  
KNeighbors_classifier = KNeighborsClassifier()  
KNeighbors_classifier.fit(X_train, y_train)



from sklearn.naive_bayes import GaussianNB
#I fit the data with GNB classifier
gnb_classifier = GaussianNB()
gnb_classifier.fit(X_train, y_train) 


#make predicitons for entropy classifier
predictionentropy = entropyclassifier.predict(X_test)

#make predicitons for gini classifier
predictiongini = giniclassifier.predict(X_test) 

#make predicitons for KNN classifier
prediction_KNN = KNeighbors_classifier.predict(X_test)

#make predicitons for NB classifier
prediction_gnb = gnb_classifier.predict(X_test)


#predictions for all classifiers
print('prediction for NB ',prediction_gnb[:10])
print('prediciton for KNN ',prediction_KNN[:10])
print('prediction for gini ',predictiongini[:10])
print('prediction for entropy ',predictionentropy[:10])
print(y_test[:10])



   
print("Confusion Matrix for entropy: ")
print( confusion_matrix(y_test, predictionentropy)) 
print("Confusion Matrix for gini: ") 
print( confusion_matrix(y_test, predictiongini))
print("Confusion Matrix for K-NN: ")
print( confusion_matrix(y_test, prediction_KNN))
print("Confusion Matrix for GNB: ")
print( confusion_matrix(y_test, prediction_gnb))

     	
print ("Accuracy for entropy : ", accuracy_score(y_test,predictionentropy)*100)
print ("Accuracy for gini : ", accuracy_score(y_test,predictiongini)*100)
print ("Accuracy for KNN : ", accuracy_score(y_test,prediction_KNN)*100)
print ("Accuracy for GNB : ", accuracy_score(y_test,prediction_gnb)*100)


print("Report for entropy : ")
print( classification_report(y_test, predictionentropy))
print("Report for gini : ")
print(classification_report(y_test, predictiongini))
print("Report for KNN : ")
print(classification_report(y_test, prediction_KNN))
print("Report for GNB : ")
print(classification_report(y_test, prediction_gnb))


import matplotlib.pyplot as plt 

# plotting a histogram 
plt.hist(y_train, color = 'black', 
        histtype = 'bar') 
  
# x-axis label 
plt.xlabel('category') 
# y-axis label 
plt.ylabel('Population') 
# plot title 
plt.title('Number of each category') 
  
# function to show the plot 
plt.show()





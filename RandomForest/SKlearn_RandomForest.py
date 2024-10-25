import random
import pandas
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier  
from sklearn.metrics import classification_report,accuracy_score,f1_score,make_scorer
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
negative = pandas.read_csv('C:\\Users\\lenovo\\Documents\\seventhSemester\\Artificial_Inteligence\\Second_Project\\traindataframes\\negative_m500_n21_k25720.csv')
positive = pandas.read_csv('C:\\Users\\lenovo\\Documents\\seventhSemester\\Artificial_Inteligence\\Second_Project\\traindataframes\\positive_m500_n21_k25720.csv')

negative=negative.drop("Unnamed: 0",axis=1)
positive=positive.drop("Unnamed: 0",axis=1)
  
features=list(negative.columns)
features.remove('rating')


y_n=negative['rating']
y_p=positive['rating']
y_train=pandas.concat([y_n, y_p], ignore_index = True)
y_train= y_train.astype('int')

negative=negative.drop("rating",axis=1)
positive=positive.drop("rating",axis=1)
x_n=negative
x_p=positive
x_train=pandas.concat([x_n, x_p], ignore_index = True)

train_accuracies = []
test_accuracies = []
precisions = []
recalls = []
f1_scores = []
trainin_data=[]
for times in range(1,11,1):
    size=random.randint(15000,len(x_train))
    print("size",size)
    x=x_train.iloc[:size,:]
    ind=x.index.values
    y=y_train.iloc[ind]
    # Splitting the dataset into training and test set
    X_train, X_test, Y_train, y_test = train_test_split(x, y, test_size=0.2)
    
    #Fitting Decision Tree classifier to the training set  
    classifier= RandomForestClassifier(n_estimators= 10, criterion="entropy")  
    classifier.fit(X_train, Y_train)  

    #Predicting the test set result  
    y_pred= classifier.predict(X_test)  
    test_accuracy=accuracy_score(y_test,y_pred)
        
    #Train predictions
    train_predictions=classifier.predict(X_train)
    train_accuracy=accuracy_score(Y_train,train_predictions)
        

    # Calculate metrics for the positive class (class 1)
    test_results=classification_report(y_test,y_pred,zero_division=1,labels=['1'],output_dict=True)
    precision=test_results['1']['precision']
    recall=test_results['1']['recall']
    f1=test_results['1']['f1-score']
    train_size=len(X_train)
    train_accuracy=train_accuracy * 100
    test_accuracy=test_accuracy * 100

    trainin_data.append(len(X_train))
    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)
        
    data=[[train_size,train_accuracy,test_accuracy,precision,recall,f1]]
    df = pandas.DataFrame(data, columns=['Train Size', 'Train Accuracy','Test Accuracy','Precision','Recall','F1 Score'])
      
    print(df)

#Sorting the data for the curves 
dictprec={}
dictrec={}
dictf1={}
dicttrain={}
dicttest={}
i=0
for x in trainin_data:
    dictprec[x]=precisions[i]
    dictrec[x]=recalls[i]
    dictf1[x]=f1_scores[i]
    dicttrain[x]=train_accuracies[i]
    dicttest[x]=test_accuracies[i]
    i+=1

myKeys = list(dictprec.keys())
myKeys.sort()
sorted_dict = {i: dictprec[i] for i in myKeys}  
sorted_recall = {i: dictrec[i] for i in myKeys} 
sorted_f1 = {i: dictf1[i] for i in myKeys} 
sorted_train = {i: dicttrain[i] for i in myKeys} 
sorted_test = {i: dicttest[i] for i in myKeys} 
print(sorted_recall)
 

# Plot precision, recall, and F1 scores
plt.figure(figsize=(10, 6))
plt.plot(sorted_dict.keys(),sorted_dict.values(), label='Precision')
plt.plot(sorted_recall.keys(), sorted_recall.values(), label='Recall')
plt.plot(sorted_f1.keys(), sorted_f1.values(), label='F1 Score')
plt.title('Metrics vs. Training Set Size')
plt.xlabel('Training Set Size')
plt.ylabel('Score')
plt.legend()
plt.show()   

# Plot accuracy curves
plt.figure(figsize=(10, 6))
plt.plot(sorted_dict.keys(),sorted_train.values(), label='Train Accuracy')
plt.plot(sorted_test.keys(),sorted_test.values(), label='Test Accuracy')
plt.title('Accuracy Curves ')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.show()

#Learning Curve
scorer = make_scorer(f1_score)
train_sizes, train_scores, test_scores = learning_curve(
                    estimator=classifier,
                    X=x_train,
                    y=y_train,
                    cv=5,
                    n_jobs=50,
                    scoring=scorer,
                    shuffle=True,
                    train_sizes = [5000, 10000,15000,17000,20000]
                )

train_mean = train_scores.mean(axis=1)
test_mean = test_scores.mean(axis=1)

plt.subplots(figsize=(10,8))
plt.plot(train_sizes, train_mean, label="train")
plt.plot(train_sizes, test_mean, label="validation")

plt.title("Learning Curve")
plt.xlabel("Training Set Size")
plt.ylabel("f1-score")
plt.legend(loc="best")

plt.show()

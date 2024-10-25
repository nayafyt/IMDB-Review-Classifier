import random
import pandas
import id3
import numpy as np
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt

from sklearn.metrics import make_scorer,f1_score,accuracy_score
from sklearn.model_selection import learning_curve
from sklearn.metrics import classification_report


class RandomForest:
    def __init__(self, n_trees, features):
        self.n_trees = n_trees
        self.features = features
        self.forest = []

    def get_params(self,deep=True):
        dict={}
        dict["n_trees"]=self.n_trees
        dict["features"]=self.features
        return dict
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    
    def score(self,y_true,y_pred):
         # Calculate F1-score
        f1 = f1_score(y_true, y_pred)
        return f1
    
    
    def fit(self, x_Train, y_Train):
        #build the random forest
        i=self.n_trees
        
        while  i>0:
            #selecting a random subset of features for each decision tree
           
            k=random.randint(60,70)     
            print("Number of features: ",k)
            max_features=k
            j=max_features
            feat=[]
            while j>0:
                feat.append(random.choice(self.features))
                j=j-1 

            #selecting a random subset of data  
            starters=x_Train[feat]
            X_train=starters.sample(frac=.8,replace=True) #80% of data with replacement
            indexes=list(X_train.index.values)
            Y_train=y_Train.loc[indexes] #y data must correspond with x data
            
            
            #implementing id3 to create each tree
            loona=id3.ID3(feat)
            tree=loona.fit(X_train.values,Y_train.values)
            self.forest.append(tree)
            print("tree is made",i)
            i=i-1    
        return self 
    
    def predict(self, X):
        '''using the trained random forest to make predictions'''
        allpredictions=[]
        #we call the id3 predict function for each tree
        for x in self.forest:
            loona=id3.ID3(self.features)
            loona.tree=x
            individual_pred=loona.predict(X.values)
            allpredictions.append(individual_pred)

        #majority voting 
        finalpred=[]
        j=0
        while j< len(allpredictions[0]):
            negs=0
            pos=0
            for x in allpredictions:
                if x[j]==1:
                    pos+=1
                else:
                    negs+=1 
            luck=random.randint(0,1)        
            if pos>negs:
                finalpred.append(1)   
            elif negs>pos:
                finalpred.append(0)   
            else:
                if luck==1:
                    finalpred.append(1)  
                else:
                    finalpred.append(0)               
            j+=1  
        return finalpred
    
    #Train the Random Forest model#
    def RandomForest_Train(self,x_train, y_train,number_of_Trees):

        # Splitting the dataset into training and test set.  
        
        size=random.randint(15000,len(x_train))
        x=x_train.iloc[:size,:]
        ind=x.index.values
        y=y_train.iloc[ind]

        x_Train, x_Test, y_Train, y_Test= train_test_split(x, y, test_size= 0.15,shuffle=True)
        
        #Train the model 
        random_forest=RandomForest(number_of_Trees,features)
        random_forest.fit(x_Train,y_Train)

        #Test predictions
        test_predictions = random_forest.predict(x_Test)
        test_accuracy=accuracy_score(y_Test,test_predictions)
        
        #Train predictions
        train_predictions=random_forest.predict(x_Train)
        train_accuracy=accuracy_score(y_Train,train_predictions)
        

        # Calculate metrics for the positive class (class 1)
        test_results=classification_report(y_Test,test_predictions,zero_division=1,labels=['1'],output_dict=True)
        precision=test_results['1']['precision']
        recall=test_results['1']['recall']
        f1=test_results['1']['f1-score']
        train_size=len(x_Train)
        train_accuracy=train_accuracy * 100
        test_accuracy=test_accuracy * 100
        
        data=[[train_size,train_accuracy,test_accuracy,precision,recall,f1]]
        df = pandas.DataFrame(data, columns=['Train Size', 'Train Accuracy','Test Accuracy','Precision','Recall','F1 Score'])
      
        print(df)
        

        
        return train_size,train_accuracy,test_accuracy,precision,recall,f1
    
###### TRAIN DATA ###### 
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
for i in range(1,11,1):
    random_forest=RandomForest(1,features)
    tr,tracc,testacc,prec,rec,f1=random_forest.RandomForest_Train(x_train,y_train,10)
    trainin_data.append(tr)
    train_accuracies.append(tracc)
    test_accuracies.append(testacc)
    precisions.append(prec)
    recalls.append(rec)
    f1_scores.append(f1)

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

# Create a scorer object 
random_forest=RandomForest(10,features)
scorer = make_scorer(random_forest.score)
train_sizes, train_scores, test_scores = learning_curve(
                    estimator=random_forest,
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


   
     
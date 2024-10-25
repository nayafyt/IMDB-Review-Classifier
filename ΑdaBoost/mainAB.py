import adaboost
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt


negative = pd.read_csv('C:\\Users\\nfyta\\Downloads\\maincsv\\negative_m500_n21_k25720.csv')
positive = pd.read_csv('C:\\Users\\nfyta\\Downloads\\maincsv\\positive_m500_n21_k25720.csv')

negative=negative.drop("Unnamed: 0",axis=1)
positive=positive.drop("Unnamed: 0",axis=1)

features=list(negative.columns)
features.remove('rating')


y_n=negative['rating']
y_p=positive['rating']
y_train=pd.concat([y_n, y_p], ignore_index = True)
y_train= y_train.astype('int')

negative=negative.drop("rating",axis=1)
positive=positive.drop("rating",axis=1)
x_n=negative
x_p=positive
x_train=pd.concat([x_n, x_p], ignore_index = True)

# Shuffle the input data
indexes = random.sample(range(len(x_train)), len(x_train))
x_train = x_train.iloc[indexes].reset_index(drop=True)
y_train = y_train.iloc[indexes].reset_index(drop=True)

# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(x_train, y_train, test_size=0.2, random_state=42)



########Τα προηγούμενα είναι κώδικας Σάρας



# I'm keeping only the values (if the word exists or not on the review)
train_features_subset = train_features.iloc[:, :]

# Iterate through each row and create a NumPy array
train_vector = [row.values for i, row in train_features_subset.iterrows()]
train_vector = np.array(train_vector)
# We keep only if the review is positive (1) or negative
train_labels_arr = np.array(train_labels)


##Test 
test_features_subset = test_features.iloc[:, :]

# Iterate through each row and create a NumPy array
test_vector = [row.values for i, row in test_features_subset.iterrows()]
test_vector = np.array(test_vector)
# We keep only if the review is positive (1) or negative
test_labels_arr = np.array(test_labels)


# Set parameters
max_train_size = len(train_vector)
step_size = int(max_train_size / 10)

# Evaluate AdaBoost with varying training set sizes
train_sizes, train_accuracies, test_accuracies, precisions, recalls, f1_scores = adaboost.evaluate_adaboost(
    train_vector, train_labels_arr, test_vector, test_labels_arr, max_train_size, step_size
)

# Plot learning curves
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_accuracies, label='Train Accuracy')
plt.plot(train_sizes, test_accuracies, label='Test Accuracy')
plt.title('Learning Curves - Accuracy')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.show()

# Plot precision, recall, and F1 scores
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, precisions, label='Precision')
plt.plot(train_sizes, recalls, label='Recall')
plt.plot(train_sizes, f1_scores, label='F1 Score')
plt.title('Metrics vs. Training Set Size')
plt.xlabel('Training Set Size')
plt.ylabel('Score')
plt.legend()
plt.show()

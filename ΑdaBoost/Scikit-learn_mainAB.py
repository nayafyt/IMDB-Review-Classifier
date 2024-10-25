import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,classification_report
import matplotlib.pyplot as plt

# Load your data
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

# Set parameters
max_train_size = len(train_features)
step_size = int(max_train_size / 10)

# Initialize lists to store results
train_sizes = []
train_accuracies = []
test_accuracies = []
precisions = []
recalls = []
f1_scores = []

# Evaluate AdaBoost with varying training set sizes
for size in range(step_size, max_train_size + 1, step_size):
    # Subset the data
    train_features_subset = train_features.iloc[:size, :]
    train_labels_subset = train_labels[:size]

    # Train AdaBoost classifier
    model = AdaBoostClassifier(n_estimators=6, random_state=42)
    model.fit(train_features_subset, train_labels_subset)

    # Make predictions
    train_predictions = model.predict(train_features_subset)
    test_predictions = model.predict(test_features)

    # Calculate metrics
    train_accuracy = accuracy_score(train_labels_subset, train_predictions)
    test_accuracy = accuracy_score(test_labels, test_predictions)
    # Calculate metrics for the negative class
    test_results= classification_report(test_predictions, test_labels,output_dict=True)
    #Με false να είναι οι αρνητικές κριτικές που αναγνώρισε σωστά (δηλαδή ο αλγόριθμος προέβλεψε ότι ήταν αρνητικές και όντως ήταν)
    precision=test_results['0']['precision']
    recall=test_results['0']['recall']
    f1=test_results['0']['f1-score']

    # Append results to lists
    train_sizes.append(size)
    train_accuracies.append(train_accuracy * 100)
    test_accuracies.append(test_accuracy * 100)
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)

    results_df = pd.DataFrame({
        'Train Size': train_sizes,
        'Train Accuracy': train_accuracies,
        'Test Accuracy': test_accuracies,
        'Precision': precisions,
        'Recall': recalls,
        'F1 Score': f1_scores
    })

    print(results_df)

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

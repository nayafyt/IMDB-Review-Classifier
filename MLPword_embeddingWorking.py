import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

def evaluate_MLP(train_vector, train_labels_arr, max_train_size, step_size):
    train_sizes = []
    train_accuracies = []
    test_accuracies = []

    precisions = []
    recalls = []
    f1_scores = []


    epochs = 2  #Δηλώνουμε τον αριθμό των εποχών/περιόδων

    # Προσθήκη λίστας για την αποθήκευση του σφάλματος σε κάθε εποχή
    train_losses_per_epoch = []
    test_losses_per_epoch = []


    #Χωρίζουμε κάθε φορά σε διαφορετικό μέρος τα train και test 
    for train_size in range(step_size, max_train_size + 1, step_size):
        # Train classifiers
        # Εκπαίδευση του μοντέλου κάθε φορά για μεγαλύτερο training_size (όμοια με adaboost)
        classifiers = model.fit(train_vector[:train_size, :], train_labels_arr[:train_size], epochs=epochs, batch_size=100, validation_split=0.2)

        # Προβλέψεις στο test set
        test_predictions = model.predict(test_features).flatten()

        # Ορίζουμε ένα κατώφλι για τις προβλέψεις
        threshold = 0.5
        binary_test_predictions = (test_predictions > threshold).astype(int)

        # Υπολογισμός των μετρικών precision, recall, και F1
        classification_rep = classification_report(test_labels, binary_test_predictions, output_dict=True)

        # Εκτύπωση των καμπυλών precision, recall, και F1
        # Mε 0 να είναι οι αρνητικές κριτικές που αναγνώρισε σωστά (δηλαδή ο αλγόριθμος προέβλεψε ότι ήταν αρνητικές και όντως ήταν)
        precision = classification_rep[str(0)]['precision'] 
        recall = classification_rep[str(0)]['recall'] 
        f1 = classification_rep[str(0)]['f1-score'] 

        train_losses_per_epoch.append(classifiers.history['loss'])
        # Εκτύπωση των αποτελεσμάτων των δεδομένων εκπαίδευσης και ελέγχου
        train_loss, train_accuracy = model.evaluate(train_features, train_labels)
        
       
        test_loss, test_accuracy = model.evaluate(test_features, test_labels)
        test_losses_per_epoch.append(test_loss)

        # Append results to lists
        train_sizes.append(train_size)
        train_accuracies.append(train_accuracy * 100)
        test_accuracies.append(test_accuracy * 100)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    # Εκτύπωση των πινάκων με τα αποτελέσματα

    results_df = pd.DataFrame({
        'Train Size': train_sizes,
        'Train Accuracy': train_accuracies,
        'Test Accuracy': test_accuracies,
        'Train Loss': train_losses_per_epoch,   
        'Test Loss': test_losses_per_epoch,
        'Precision': precisions,
        'Recall': recalls,
        'F1 Score': f1_scores
    })

    print(results_df)

    return train_sizes, train_accuracies, test_accuracies,train_losses_per_epoch, test_losses_per_epoch, precisions, recalls, f1_scores


# Φόρτωση των δεδομένων, όπως την έχουμε σε κάθε συνάρτηση 
negative = pd.read_csv('C:\\Users\\nfyta\\Desktop\\AIproj2\\negative_m500_n21_k25720.csv')
positive = pd.read_csv('C:\\Users\\nfyta\\Desktop\\AIproj2\\positive_m500_n21_k25720.csv')

negative = negative.drop("Unnamed: 0", axis=1)
positive = positive.drop("Unnamed: 0", axis=1)

features = list(negative.columns)
features.remove('rating')

y_n = negative['rating']
y_p = positive['rating']
y_train = pd.concat([y_n, y_p], ignore_index=True)
y_train = y_train.astype('int')

negative = negative.drop("rating", axis=1)
positive = positive.drop("rating", axis=1)
x_n = negative
x_p = positive
x_train = pd.concat([x_n, x_p], ignore_index=True)

# Shuffle the input data
indexes = random.sample(range(len(x_train)), len(x_train))
x_train = x_train.iloc[indexes].reset_index(drop=True)
y_train = y_train.iloc[indexes].reset_index(drop=True)


# Split the data into training and testing sets το 20% των δεδομένων γίνονται test-data
train_features, test_features, train_labels, test_labels = train_test_split(x_train, y_train, test_size=0.2,
                                                                            random_state=42)

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


#Ίδιος κώδικας με τα προηγούμενα


# Υποθέτουμε ότι οι ενθέσεις έχουν διαστάσεις 100, δηλαδή μεταγενέστερα τα διανύσματα
embedding_dim = 100

# Δημιουργία του μοντέλου
model = Sequential()
model.add(Embedding(input_dim=len(features), output_dim=embedding_dim, input_length=train_features.shape[1]))
model.add(Flatten())

#Δημιουργούμε 3 στρώσεις με 64, 32 και τέλος 1 νευρώνες
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Σύνθεση του μοντέλου, χρησιμοποιούμε συνάρτηση βελτιστοποίησης την Adam, για τον υπολογισμό του σφάλματος την 
#binary_coressentropy και ως μετρική σύγκριση την ακρίβεια.
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

epochs = 2  #Δηλώνουμε τον αριθμό των εποχών/περιόδων

# Evaluate MLP with varying training set sizes
train_sizes, train_accuracies, test_accuracies,train_losses_per_epoch, test_losses_per_epoch, precisions, recalls, f1_scores = evaluate_MLP(
    train_vector, train_labels_arr, max_train_size, step_size
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

# Plot loss curves per epochs
plt.figure(figsize=(10, 6))
for i in range(epochs):
    plt.plot(train_sizes, [loss[i] for loss in train_losses_per_epoch], label=f'Train Loss (Epoch {i+1})', linestyle='--')
plt.plot(train_sizes, test_losses_per_epoch, label='Test Loss', linestyle='-')
plt.title('Learning Curves - Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
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



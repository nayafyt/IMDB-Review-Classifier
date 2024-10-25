import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,classification_report

#Δημιουργούμε κάθε φορά ένα δέντρο απόφασης βάθους 1 
class Stump:

    def __init__(self):
        self.word_index = None #Εξετάζουμε μία μία λέξη
        self.rating_value = 1 #αν θεωρείται η λέξη θετική (1) ή αρνητική (0)
        self.alpha = None #βάρος του stump (όσο καλύτερα τα πάει (πιο "καλυπτική") τόσο μεγαλύτερο βάρος)

    def create_stump(self, reviews):

        N = reviews.shape[0] #πλήθος λέξεων

        #Εξετάζουμε κάθε φορά άλλη λέξη 
        best_column = reviews[:, self.word_index]

        #Δημιουργούμε τα δύο παιδιά με υπόθεση βασισμένη στο rating που έχει
        hypothesis = np.ones(N)
        if self.rating_value == 1:
            hypothesis[best_column == 0] = 0
        else:
            hypothesis[best_column == 1] = 0

        return hypothesis

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)


def train(reviews, values, n_words, n_stumps) :

    #Θέτουμε τα όρια με βάση τις κριτικές (Ν πλήθους) 
    N = reviews.shape[0]
    #Πλήθος λέξεων (στήλων)
    n_words = n_words
    #Πλήθος stump
    M = n_stumps

    #Αρχικοποίηση βαρών (ισοβαρής όσο 1/πλήθος_στοιχείων)
    weights = np.full(N, 1 / N)

    #Αρχικοποίηση για διατήρηση της κατάστασης ανά stump
    classifiers = np.empty(M, dtype=Stump)

    #Δημιουργόυμε και εξετάζουμε το κάθε stump ανεξάρτητα μεταδίδοντας από το ένα στο άλλο το σφάλμα
    for k in range(M):

        stump = Stump()

        #Αρχικοποιούμε τα σφάλματα ανά λέξη
        wordserr = []

        #Εξετάζουμε μια-μια λέξη
        for word_column in range(n_words):
            
            word_vector = reviews[:, word_column]

            rating_value = 1
            #Αρχικοποιούμε όλες τις λέξεις να θεωρούνται θετικές 
            hypothesis = np.ones(N)
            
            hypothesis[word_vector != rating_value] = 0

            #Υπολογίζουμε το σφάλμα
            error = sum(weights[hypothesis != values])

            #Άμα έχουμε πολύ μεγάλο σφάλμα σημαίνει ότι δεν ισχύει σίγουρα η υπόθεση μας άρα βάζουμε την αντίθετη
            if error > 0.5:
                rating_value = 0
                error = 1 - error

            #Προσθέτουμε το σφάλμα και την τιμή που καταλήξαμε
            wordserr.append((error, rating_value))

        #Κρατάμε πάντα το πρώτο στοιχείο δηλαδή τα errors 
        errors = list(map(lambda x: x[0], wordserr))
        #Κρατάμε τις λέξεις με το μικρότερο σφάλμα, οι οποίες είναι αυτές που "καλύφθηκαν" από το stump 
        stump.word_index = errors.index(min(errors))
        #Κρατάμε το error το λέξεων που θεωρήθηκαν "καλυμμένες" για να υπολογίσουμε το βάρος του stump
        t_error, stump.rating_value = wordserr[stump.word_index]

        #Υπολογισμός του βάρους
        stump.alpha = 0.5 * np.log((1 - t_error) / t_error)

        #Κρατάμε και την υπόθεση του rating της συγκεκριμένης λέξης που προήλθε από το stump
        hypothesis = stump.create_stump(reviews)

        #Υπολογίζουμε τα νέα βάρη, μειώνουμε το βάρος αυτών που το rating όπως υπολογίστηκε από το stump με το πραγματικό
        #ήταν ίδιο (άρα θεωρούνται ότι καλύφθηκαν από το stump), τα άλλα τα αφήνουμε ίδια
        for i in range(N):
            if hypothesis[i] == values[i]:
                weights[i] *= t_error/(1 - t_error)
        
        #Κανονικοποιούμε τα βάρη, αφού θέλουμε πάντα το άθροισμα τους να είναι 1
        weights /= np.sum(weights)

        #Αποθηκεύουμε την κατάσταση κάθε stump
        classifiers[k] = stump

    return classifiers


def predict(reviews, classifiers):

    #Με βάση το βάρος των stump, υπολογίζουμε την συνολική κάλυψη του συγκεκριμένου αλγορίθμου 
    a_sum = 0
    for stump in classifiers:
        a_sum += stump.alpha
    threshold = a_sum / 2

    predictions = [stump.alpha * stump.create_stump(reviews) for stump in classifiers]
    predictions_sums = np.sum(predictions, axis = 0)
    final_prediction = predictions_sums > threshold

    return final_prediction


#Μέθοδος για τον υπολογισμό ακρίβειας (precision), ανάκλησης (recall), F1
def calculate_metrics(predictions, labels):
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    return precision, recall, f1

# Εξετάζουμε την adaboost επαναληπτικά, για να εξάγουμε αρκετές πληροφορίες για την δημιουργία καμπύλων και των άλλων ζητουμένων
def evaluate_adaboost(train_vector, train_labels_arr, test_vector, test_labels_arr, max_train_size, step_size):
    train_sizes = []
    train_accuracies = []
    test_accuracies = []
    precisions = []
    recalls = []
    f1_scores = []

    #Χωρίζουμε κάθε φορά σε διαφορετικό μέρος τα train και test 
    for train_size in range(step_size, max_train_size + 1, step_size):
        # Train classifiers
        classifiers =train(train_vector[:train_size, :], train_labels_arr[:train_size], train_vector.shape[1], 6)

        # Train predictions
        train_predictions = predict(train_vector[:train_size, :], classifiers)
        train_accuracy = accuracy_score(train_labels_arr[:train_size], train_predictions)
        
        # Test predictions
        test_predictions = predict(test_vector, classifiers)
        test_accuracy = accuracy_score(test_labels_arr, test_predictions)

        # Calculate metrics for the negative class
        test_results= classification_report(test_predictions, test_labels_arr,output_dict=True)
        #Με false να είναι οι αρνητικές κριτικές που αναγνώρισε σωστά (δηλαδή ο αλγόριθμος προέβλεψε ότι ήταν αρνητικές και όντως ήταν)
        precision=test_results['False']['precision']
        recall=test_results['False']['recall']
        f1=test_results['False']['f1-score']
       
 

        
        # Append results to lists
        train_sizes.append(train_size)
        train_accuracies.append(train_accuracy * 100)
        test_accuracies.append(test_accuracy * 100)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    # Εκτύπωση των πινάκων με τα αποτελέσματα
    # Εκτύπωση των πινάκων με τα αποτελέσματα
    results_df = pd.DataFrame({
        'Train Size': train_sizes,
        'Train Accuracy': train_accuracies,
        'Test Accuracy': test_accuracies,
        'Precision': precisions,
        'Recall': recalls,
        'F1 Score': f1_scores
    })

    print(results_df)

    return train_sizes, train_accuracies, test_accuracies, precisions, recalls, f1_scores




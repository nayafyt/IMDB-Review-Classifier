import glob
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from gensim.models import Word2Vec

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
        classifiers=model.fit(train_vector[:train_size, :], train_labels_arr[:train_size], epochs=epochs, batch_size=100, validation_split=0.2)

        # Προβλέψεις στο test set
        test_predictions = model.predict(x_test).flatten()

        # Ορίζουμε ένα κατώφλι για τις προβλέψεις
        threshold = 0.5
        binary_test_predictions = (test_predictions > threshold).astype(int)

        # Υπολογισμός των μετρικών precision, recall, και F1
        classification_rep = classification_report(y_test, binary_test_predictions, output_dict=True)

        # Εκτύπωση των καμπυλών precision, recall, και F1
        # Mε 0 να είναι οι αρνητικές κριτικές που αναγνώρισε σωστά (δηλαδή ο αλγόριθμος προέβλεψε ότι ήταν αρνητικές και όντως ήταν)
        precision = classification_rep[str(0)]['precision'] 
        recall = classification_rep[str(0)]['recall'] 
        f1 = classification_rep[str(0)]['f1-score'] 

        train_losses_per_epoch.append(classifiers.history['loss'])
        # Εκτύπωση των αποτελεσμάτων των δεδομένων εκπαίδευσης και ελέγχου
        train_loss, train_accuracy = model.evaluate(x_train, y_train)
        
       
        test_loss, test_accuracy = model.evaluate(x_test, y_test)
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

# Define a generator function to load data in batches
def data_generator(batch_size, word_embeddings, labels):
    num_samples = len(labels)
    start_index = 0
    while True:
        if start_index >= num_samples:
            start_index = 0
        end_index = min(start_index + batch_size, num_samples)
        batch_x = np.array(word_embeddings[start_index:end_index])
        batch_y = np.array(labels[start_index:end_index])
        start_index = end_index
        yield batch_x, batch_y

# Επεξεργασία των κειμένων
path = input("Insert the folder path containing labeledBow files: ")
positive_files = glob.glob(os.path.join(path, 'pos*.txt'))
negative_files = glob.glob(os.path.join(path, 'neg*.txt'))
path=input ("Insert the file path of labeledBow: ")
freq=open(path,'r')
text=freq.read()
content=text.splitlines()

#ftiaxnoume dictionary gia tin synoliki syxnotita
dict={}
i=0
while i<89527 : #89527 is len(vocabs)
    dict[i]=0
    i+=1

i=0    
while i<len(content):
    numbers=content[i].split()
    j=1
    while j<len(numbers):
        x=numbers[j].split(":")
        key=int(x[0])
        times=int(x[1])
        dict[key]=dict[key]+times
        j+=1
    i+=1    



path2=input("Insert the file path of imdb.vocab: ")
f = open(path2, 'rb')
vocabs = [line.decode('utf8')[:-1] for line in f]
f.close()

#afairoume tis oudeteres-axristes lekseis me vash to polarity tous
path3=input("Insert the file path of imdbEr: ")
polF=open(path3)
polar=polF.read()
polarity=polar.splitlines()

keys=[] #initialize list of keys of most frequent words
i=0
while i<len(vocabs):
    keys.append(i)
    i+=1

redundant=[] #keys that must be erased
for x in keys:
    if abs(float(polarity[x]))<0.7: #thewrw oti katw apo 0.7 einai oudeteri leksi xwris aksia
        redundant.append(x)
#to parakatw ginetai gia na afairethoun swsta ola ta redundant kleidia giati ta evgaze ana dyo mesa sto panw if epeidi allazame thn keys       
i=0  
newkeys=[]      
while i<len(keys):
    if keys[i] not in redundant:
        newkeys.append(i)  
    i+=1      
polF.close()    

print("length ",len(newkeys))

 
#Hyper-parameters
m=int(input("Insert a value for m: "))
n=int(input("Insert a value for n: "))
k=int(input("Insert a value for k: "))
#prepei m=totalwords-n-k
while m!=(len(newkeys)-n-k):
    print("You inserted wrong data.This rule must be met : m= ",len(newkeys),"-n-k")
    m=int(input("Insert a value for m: "))
    n=int(input("Insert a value for n: "))
    k=int(input("Insert a value for k: "))

#af  airw apo to copydict ose den yparxoun sto newkeys 
i=0     
copydict={}   
while i<len(newkeys):
    copydict[newkeys[i]]=dict[newkeys[i]]
    i+=1


removal=[]
maximum= max(copydict.values())
while n>0 : #prwta paraleipoume tis n pio syxnes 
    for x in copydict:
        if maximum==copydict[x]:
            n=n-1
            if n>=0:
                removal.append(x)
    for x in removal:
        if x in copydict:
            copydict.pop(x) #we remove the checked element
    maximum= max(copydict.values())
    
    
epeksergasmena=[] #enapomeinanta kleidia
for x in newkeys:
    if x not in removal:
        epeksergasmena.append(x)    
print("Keys after n removal ",epeksergasmena,"length ",len(epeksergasmena))    

removal.clear()
minimum=min(copydict.values())
while k>0: #meta afairoume tis k pio spanies 
    for x in copydict:
        if minimum==copydict[x]:
            k=k-1
            if k>=0:
                removal.append(x)
    for x in removal:
        if x in copydict:
            copydict.pop(x) #we remove the checked element
    minimum= min(copydict.values())

kleidia=[] #enapomeinanta kleidia
for x in epeksergasmena:
    if x not in removal:
        kleidia.append(x)        

print("Keys after k removal ",kleidia,"length ",len(kleidia)) 


freq.close()    


mostFrequentWords={}
for x in kleidia:
    mostFrequentWords[vocabs[x]]=0

#lista me ola ta thetika reviews 
previews=[]

#metatropi keimenou se vector 
path4=input("Insert the positive reviews folder path: ")
os.chdir(path4)
pos_dir_list= glob.glob('*.txt')
kl=0
for file in pos_dir_list:
    with open(file, 'r',encoding='cp437') as f:
        text=f.read()
        list=text.lower().split(" ")
        i=0
        while i<len(list):
            list[i]=list[i].lower()
            if "." in list[i]: #removing the fullstop from words
                list[i]=list[i].replace(".","")
            if list[i].endswith(","): #removing the comma from words
                ind=list[i].index(",")
                new=list[i][:ind]
                list[i]=new
            if list[i].endswith(":"): #removing the : from words
                ind=list[i].index(":")
                new=list[i][:ind]
                list[i]=new
            if list[i].endswith("\'") or list[i].startswith("\'"): #removing the ' from words
                a = list[i]
                a = a.replace("\'","")
                list[i]=a  
            if list[i].endswith(")") or list[i].startswith(")") : #removing the ) from words
                a = list[i]
                a = a.replace(")","")
                list[i]=a   
            if list[i].startswith("(") or list[i].endswith("("): #removing the ( from words
                a = list[i]
                a = a.replace("(","")
                list[i]=a        
            if list[i].endswith("?"): #removing the ? from words
                ind=list[i].index("?")
                new=list[i][:ind]
                list[i]=new   
            if list[i].endswith("!"): #removing the ! from words
                ind=list[i].index("!")
                new=list[i][:ind]
                list[i]=new 
            if list[i].startswith("/"): #removing the / from words
                a = list[i]
                a = a.replace("/","")
                list[i]=a 
            if list[i].startswith(">"): #removing the > from words
                a = list[i]
                a = a.replace(">","")
                list[i]=a       
            if list[i].endswith("<br"):
                a=list[i]
                a=a.replace("<br","")
                list[i]=a  
            if "(" in list[i] :
                a=list[i].replace("("," ")
                words=a.split()
                list[i]=words[0]
                if len(words)==2:
                    list.append(words[1])
                if list[i].endswith(")") or list[i].startswith(")") : #removing the ) from words
                    a = list[i]
                    a = a.replace(")","")
                    list[i]=a   
                if list[i].startswith("(") or list[i].endswith("("): #removing the ( from words
                    a = list[i]
                    a = a.replace("(","")
                    list[i]=a         
            if ")" in list[i] :
                a=list[i].replace(")"," ")
                words=a.split()
                list[i]=words[0]
                if len(words)==2:
                    list.append(words[1]) 
                if list[i].endswith(")") or list[i].startswith(")") : #removing the ) from words
                    a = list[i]
                    a = a.replace(")","")
                    list[i]=a   
                if list[i].startswith("(") or list[i].endswith("("): #removing the ( from words
                    a = list[i]
                    a = a.replace("(","")
                    list[i]=a                   
            #removing the "" from words
            a = list[i]
            a = a.replace('"', '')
            list[i]=a             
            if list[i]=="":
                list.pop(i) 
                i=i-1     
            i+=1
        previews.append(list)

nreviews=[]
path5=input("Insert the negative reviews folder path: ")
os.chdir(path5)
neg_dir_list= glob.glob('*.txt')
kl=0
for file in neg_dir_list:
    with open(file, 'r',encoding='cp437') as f:
        text=f.read()
        list=text.lower().split(" ")
        i=0
        while i<len(list):
            list[i]=list[i].lower()
            if "." in list[i]: #removing the fullstop from words
                list[i]=list[i].replace(".","")
            if list[i].endswith(","): #removing the comma from words
                ind=list[i].index(",")
                new=list[i][:ind]
                list[i]=new
            if list[i].endswith(":"): #removing the : from words
                ind=list[i].index(":")
                new=list[i][:ind]
                list[i]=new
            if list[i].endswith("\'") or list[i].startswith("\'"): #removing the ' from words
                a = list[i]
                a = a.replace("\'","")
                list[i]=a  
            if list[i].endswith(")") or list[i].startswith(")") : #removing the ) from words
                a = list[i]
                a = a.replace(")","")
                list[i]=a   
            if list[i].startswith("(") or list[i].endswith("("): #removing the ( from words
                a = list[i]
                a = a.replace("(","")
                list[i]=a        
            if list[i].endswith("?"): #removing the ? from words
                ind=list[i].index("?")
                new=list[i][:ind]
                list[i]=new   
            if list[i].endswith("!"): #removing the ! from words
                ind=list[i].index("!")
                new=list[i][:ind]
                list[i]=new 
            if list[i].startswith("/"): #removing the / from words
                a = list[i]
                a = a.replace("/","")
                list[i]=a 
            if list[i].startswith(">"): #removing the > from words
                a = list[i]
                a = a.replace(">","")
                list[i]=a       
            if list[i].endswith("<br"):
                a=list[i]
                a=a.replace("<br","")
                list[i]=a  
            if "(" in list[i] :
                a=list[i].replace("("," ")
                words=a.split()
                list[i]=words[0]
                if len(words)==2:
                    list.append(words[1])
                if list[i].endswith(")") or list[i].startswith(")") : #removing the ) from words
                    a = list[i]
                    a = a.replace(")","")
                    list[i]=a   
                if list[i].startswith("(") or list[i].endswith("("): #removing the ( from words
                    a = list[i]
                    a = a.replace("(","")
                    list[i]=a         
            if ")" in list[i] :
                a=list[i].replace(")"," ")
                words=a.split()
                list[i]=words[0]
                if len(words)==2:
                    list.append(words[1]) 
                if list[i].endswith(")") or list[i].startswith(")") : #removing the ) from words
                    a = list[i]
                    a = a.replace(")","")
                    list[i]=a   
                if list[i].startswith("(") or list[i].endswith("("): #removing the ( from words
                    a = list[i]
                    a = a.replace("(","")
                    list[i]=a                   
            #removing the "" from words
            a = list[i]
            a = a.replace('"', '')
            list[i]=a             
            if list[i]=="":
                list.pop(i) 
                i=i-1     
            i+=1
        nreviews.append(list)

# Όλα τα κείμενα
all_reviews = previews + nreviews

# Δημιουργία παραθύρων κειμένων
window_size = 3
windows = []

for review in all_reviews:
    for i in range(len(review)):
        window = review[max(0, i - window_size):min(len(review), i + window_size + 1)]
        windows.append(window)

# Επιλέγουμε τισ 1000 πιο συχνές λέξεις
num_most_common_words = len(mostFrequentWords)       
if len(mostFrequentWords)>1000:
    num_most_common_words = 1000

# Αποθηκεύουμε το νέο σύνολο πιο συχνών λέξεων
most_common_words = sorted(mostFrequentWords, key=mostFrequentWords.get, reverse=True)[:num_most_common_words]

# Δημιουργία ενός νέου λεξικού με τις πιο συχνές λέξεις
new_mostFrequentWords = {word: mostFrequentWords[word] for word in most_common_words}
# Δημιουργία κεντροειδών ενθέσεων λέξεων
word_embeddings = []
for window in windows:
    embedding = np.zeros(len(new_mostFrequentWords))  # Αρχικοποίηση μηδενικού διανύσματος
    count = 0
    for word in window:
        if word in new_mostFrequentWords:
            embedding += new_mostFrequentWords[word]  # Προσθήκη του διανύσματος της λέξης στο συνολικό διάνυσμα
            count += 1
    if count > 0:
        embedding /= count  # Υπολογισμός του μέσου όρου
    word_embeddings.append(embedding)

labels = np.array([1] * len(previews) + [0] * len(nreviews))
sample_size = 1000  # Πλήθος των δειγμάτων που θέλετε να χρησιμοποιήσετε
word_embeddings_sampled = np.array(word_embeddings)[:sample_size]
labels_sampled = np.array(labels)[:sample_size]
# Split the data into training τα οποία είναι τα word_embeddings and testing sets το 20% των δεδομένων γίνονται test-data
x_train,x_test, y_train, y_test= train_test_split(np.array(word_embeddings_sampled), labels_sampled, test_size=0.2, random_state=42)


# Set parameters
embedding_dim = word_embeddings.shape[1]  # Διανυσματικό μέγεθος των embeddings
max_train_size = len(x_train)
step_size = int(max_train_size / 10)
# Δημιουργία του μοντέλου
#Δημιουργούμε 3 στρώσεις με 64, 32 και τέλος 1 νευρώνες
# Δημιουργία του MLP μοντέλου
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(len(mostFrequentWords),)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Σύνθεση του μοντέλου, χρησιμοποιούμε συνάρτηση βελτιστοποίησης την Adam, για τον υπολογισμό του σφάλματος την 
#binary_coressentropy και ως μετρική σύγκριση την ακρίβεια.
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

epochs = 2  #Δηλώνουμε τον αριθμό των εποχών/περιόδων

# Evaluate MLP with varying training set sizes
train_sizes, train_accuracies, test_accuracies,train_losses_per_epoch, test_losses_per_epoch, precisions, recalls, f1_scores = evaluate_MLP(
    x_train, y_train, max_train_size, step_size
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



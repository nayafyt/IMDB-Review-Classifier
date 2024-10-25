import glob
import pandas as pd 
import os
import numpy as np
import copy

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
print(mostFrequentWords)
print("len toy frequent ",len(mostFrequentWords.keys()))       

#lista me ola ta thetika reviews 
reviews=[]

#metatropi keimenou se vector 
path4=input("Insert the positive reviews folder path: ")
os.chdir(path4)
pos_dir_list= glob.glob('*.txt')
kl=0
for file in pos_dir_list:
    with open(file, 'r',encoding='cp437') as f:
        text=f.read()
        list=text.split(" ")
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
        kl+=1    
        copyfr=mostFrequentWords.copy()  
        for x in list:
            if x in mostFrequentWords.keys():
                copyfr[x]=1
        if kl==1:
            dataframe=pd.DataFrame(copyfr,index=[0]) 
        else:
            new=pd.DataFrame(copyfr,index=[0])   
            dataframe=pd.concat([dataframe,new],axis=0)

static_value = '1'
dataframe['rating'] = static_value                    
dataframe.to_csv(r"C:\Users\lenovo\Documents\seventhSemester\Artificial_Inteligence\Second_Project\positive_m500_n21_k25720.csv")

nreviews=[]
path5=input("Insert the negative reviews folder path: ")
os.chdir(path5)
neg_dir_list= glob.glob('*.txt')
kl=0
for file in neg_dir_list:
    with open(file, 'r',encoding='cp437') as f:
        text=f.read()
        list=text.split(" ")
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
        kl+=1    
        copyfr=mostFrequentWords.copy()    
        for x in list:
            if x in mostFrequentWords.keys():
                copyfr[x]=1
        if kl==1:
            dataframe=pd.DataFrame(copyfr,index=[0]) 
        else:
            new=pd.DataFrame(copyfr,index=[0])   
            dataframe=pd.concat([dataframe,new],axis=0)
static_value = '0'
dataframe['rating'] = static_value
dataframe.to_csv(r"C:\Users\lenovo\Documents\seventhSemester\Artificial_Inteligence\Second_Project\negative_m500_n21_k25720.csv")   
        
        



   



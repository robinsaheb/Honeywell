import time
import subprocess
from os import listdir
from os.path import isfile, join
import sklearn.datasets
import nltk.stem
import numpy as np
import pandas as pd
from stop_words import get_stop_words
import re
import scipy as sp
import PyPDF2
import docx
import sys
import os
import os.path
import string
import docx
import docx2txt
from bs4 import BeautifulSoup

from sklearn.feature_extraction.text import TfidfVectorizer

stop_words = get_stop_words('english')
english_stemmer = nltk.stem.SnowballStemmer('english')

def file_operation():
    data = []
    mypath = "Users/sahebsingh/Desktop/Projects/Honeywell/Honeywell" #Path to folder/Directory
    list_files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    for k in range(len(list_files)):
        document = docx.Document(list_files[k])
    print (document.paragraphs)

file_operation()

def getPdf(file):
    data = []
    file = open(file, 'rb')
    pdfReader = PyPDF2.PdfFileReader(file)
    page_number = pdfReader.numPages
    for i in range(0, page_number):
        pageObj = pdfReader.getPage(i)
        data.append(pageObj.extractText())
        
    return '\n'.join(data)

def getDocx(filename):
    my_text = docx2txt.process(filename)
    
    return np.array(my_text)

data1 = getDocx('2.docx')

def getText(filename):
    doc = docx.Document(filename)
    fullText = []
    for para in doc.paragraphs:
        fullText.append(para.text)
    return '\n'.join(fullText)

def getText1(filename):
    from docx import Document
    document = Document(filename)
    for para in document.paragraphs:
        print(para.text)

data3 = getText1('2.docx')
data = getPdf('2.pdf')
data2 = getText('2.docx')

def review_to_word(raw_text,remove_stopword=False):
    #1 Remove all the HTML marks 
    review_text=BeautifulSoup(raw_text,"lxml").get_text()
    #2 Remove everything except numbers
    review_text=re.sub("[^a-zA-A]"," ",review_text)
    #3 Convert them to lower letters and split
    words=review_text.lower().split()
    #4 Option of removing stop words if they want to.
    if remove_stopword==False:
        stops=set(stop_words)
        words=[w for w in words if not w in stops]
    #5 Return Words
    return(words)

review_to_word(data2)
#Taking only Training Data.
train_data = sklearn.datasets.fetch_20newsgroups(subset="train")

print(len(train_data.filenames))
#print(train_data.target_names)                                                    

#Now we will load test data. 
test_data = sklearn.datasets.fetch_20newsgroups(subset = "test")
print(len(test_data.filenames))
#print(test_data.target_names)

""" Clustering Posts """
#This data is noisy if we run data as it is it will give UnicodeDecodeError.
#For this we will use StemmedTfidfVectorizer.

class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(TfidfVectorizer, self).build_analyzer()
        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))

vectorizer = StemmedTfidfVectorizer(min_df=10, max_df=0.5,
                        stop_words='english', decode_error='ignore'
                                    )

x_transform = vectorizer.fit_transform(train_data.data)#Fitting all the data.
num_samples, num_features = x_transform.shape
print("There are %d number of samples and %d number of features" 
        %(num_samples, num_features))

""" We will apply K-Means and keep the number of clusters as 50  """
num_clusters = 50  

from sklearn.cluster import KMeans   

#Initializing KMeans.
km = KMeans(n_clusters = num_clusters, init = 'random', n_init = 1,
            verbose = 1, random_state = 1)

#Using KMeans on x_transform.
km.fit_transform(x_transform)

post = """Disk drive problems. Hi, I have a problem with my hard disk.
After 1 year it is working only sporadically now.
I tried to format it, but now it doesn't boot any more.
Any ideas? Thanks.
"""

new_post_vec = vectorizer.transform([post])
new_post_label = km.predict(new_post_vec)[0]#We will predict it's cluster 
print(new_post_label)                                            #using km.predcit.
# Comparing all 
similar_indices = (km.labels_ == new_post_label).nonzero()[0]

similar = []
for i in similar_indices:
    dist = sp.linalg.norm((new_post_vec - x_transform[i]).toarray())
    similar.append((dist, train_data.data[i]))

similar = sorted(similar) 
print(len(similar))
show_at_1 = similar[0]
show_at_2 = similar[int(len(similar) / 10)]
show_at_3 = similar[int(len(similar) / 2)]

print("=== #1 ===")
print(show_at_1)
print()

print("=== #2 ===")
print(show_at_2)
print()

print("=== #3 ===")
print(show_at_3)
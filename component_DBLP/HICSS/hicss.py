# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 19:12:20 2017

@author: secoder
"""
import io
import hashlib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import matplotlib.pyplot as plt
import numpy as np
import nltk
from nltk.tokenize import RegexpTokenizer
from sklearn.decomposition import PCA

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from wordcloudit import wordcloudit


def featurenamebycount(a,b, hist, featurenames):
    featurename=[]
    mask = (hist >=a) & (hist<=b)
    index =np.where(mask)
    index =np.array(index)
    index = index.flatten()

    for i in index:
        featurename.append(featurenames[i])
    return featurename
        
def showfeaturenames(v,featurenames):
    names = []
    idx = np.where(v>0)
    idx = np.array(idx)
    idx = idx.flatten()
    for i in idx:
        names.append(featurenames[i])
    return names
    
"""
show titles in the same cluster
"""
def showtitles(labels, nth, alltitles):
    titles = []
    idx = np.where(labels == nth)
    idx = np.array(idx)
    idx = idx.flatten()
    for i in idx:
        titles.append(alltitles[i])
        
    return titles
    
def digstring(s):
    for i in s:
        if i.isdigit():
            return True
    return False

# remove punctuation
tokenizer = RegexpTokenizer(r'\w+')

#titledict={}
rawtitles = []
titles = []
newtitles = []
sw = set(nltk.corpus.stopwords.words('english'))

#filename = './HICSS_titles_test.txt'
filename = './HICSS_titles.txt'


with io.open(filename,'r',encoding='utf8') as f:
    #text = f.read()
    for line in f:
        #hash_object = hashlib.md5(line.encode('utf-8'))
        #titledict[hash_object.hexdigest()]=newline
        #titledict[hash_object.hexdigest()] = [w for w in line.split() if w.lower() not in sw]
        
        rawtitles.append(line)
        newline=tokenizer.tokenize(line)
        # collect all the words except digtals and stopwords
        newline= ' '.join([w for w in newline if (w.lower() not in sw) & ~(digstring(w))])
        titles.append(newline)


#print len(text)
#print titledict

#alltitles = titledict.values()
alltitles =titles

#print alltitles

# remove words only occuring in one document or 95% of the documents
vectorizer = CountVectorizer(max_df=0.95, min_df=1,stop_words='english')

X = vectorizer.fit_transform(alltitles)

analyze = vectorizer.build_analyzer()



Xarray = X.toarray()
hist = sum(Xarray)

plt.plot(hist)

transformer = TfidfTransformer()

tfidf = transformer.fit_transform(Xarray)

featurenames = vectorizer.get_feature_names()
#print vectorizer.get_feature_names()


#bigram_vectorizer = CountVectorizer(ngram_range=(1, 2),
#                                    token_pattern=r'\b\w+\b', min_df=1)
                                    
#Y = bigram_vectorizer.fit_transform(alltitles)
                                    
y = []
        
for i in range(hist.max()):                                                                                                    
    y.append(len(featurenamebycount(i,hist.max(), hist, featurenames)))
    
plt.figure()
plt.plot(range(hist.max()),y)


# do the clustering

# number of clusters
n = 10 

km=KMeans(n_clusters=n, init='k-means++',n_init=10, verbose=1)
km.fit(tfidf)

# show the word cloud of the first cluster
wordcloudit(' '.join(showtitles(km.labels_, 0, rawtitles)))


### use DBSCAN to do the clustering
### the result not good as Kmeans, need to tune parameters

#db = DBSCAN(eps=0.8, min_samples=5).fit(tfidf)
#wordcloudit(' '.join(showtitles(db.labels_, 0, rawtitles)))
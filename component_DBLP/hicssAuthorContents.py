# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 14:58:35 2017

@author: secoder
"""

import io
import nltk
from nltk.tokenize import RegexpTokenizer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from wordcloudit import wordcloudit
import matplotlib.pyplot as plt

import numpy as np


import itertools
import codecs

"""
radius of the cluster
"""
def radiusofcluster(labels, nth, dismatrix):
    idx = np.where(labels == nth)[0]

    dis = dismatrix[idx,nth]
    
    mindis = min(dis)
    maxdis = max(dis)
    
    radius = maxdis
    
    return [mindis, maxdis, radius]
        

"""
show contents in the same cluster
"""
def showcontents(labels, nth, allcontents):
    contents = []
    idx = np.where(labels == nth)
    idx = np.array(idx)
    idx = idx.flatten()
    for i in idx:
        contents.append(allcontents[i])
        
    return contents

"""
check if there is digtial in the string
"""
def digstring(s):
    for i in s:
        if i.isdigit():
            return True
    return False
    
"""
compute the distance between two points a and b
"""
def distance(a,b):
    a = np.array(a);
    b = np.array(b);
    return np.sqrt(sum(np.square(a - b)))
    
"""
"""
def updatecoauthornetwork(net,authors,namelist):
    #[r,c] = net.shape
    end = len(authors)
    
    # list to save the postions(index) of the names in the authors[]
    # in order to mark the co-authorship in the network(2d array)
    pos = []
    
    i = 0
    for name in namelist:
        if name not in authors:
            # new author, need to extend the network by 1
            #padc = np.zeros((r,1))
            #padr = np.zeros((1,c+1))
            #net = np.concatenate((net,padc), axis=1)
            #net = np.concatenate((net,padr), axis=0)
            
            #[r,c] = net.shape
    
            # increase the number of occurrence for him/her self
            net[end+i,end+i] = net[end+i,end+i] + 1
            
            # will add the new author to the authors[] list
            # so its position(indx) in the authors[] will be end+i
            pos.append(end+i)
            i = i + 1
        else:
            idx = authors.index(name)
            net[idx,idx] = net[idx,idx] + 1
            pos.append(idx)
            
    # update the network 
    # https://docs.python.org/2/library/itertools.html
    c=list(itertools.permutations(pos, 2))
    
    for pair in c:
        net[pair] = net[pair] + 1
                    
    return net
    
    

tokenizer = RegexpTokenizer(r'\w+')

#titledict={}
rawtitles = []
titles = []
newtitles = []
sw = set(nltk.corpus.stopwords.words('english'))


filename = './HICSS_titles.txt'

"""
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
"""

# use codecs        
f = codecs.open(filename,'r','utf-8')
for line in f:   
    rawtitles.append(line)
    newline=tokenizer.tokenize(line)
    # collect all the words except digtals and stopwords
    newline= ' '.join([w for w in newline if (w.lower() not in sw) & ~(digstring(w))])
    titles.append(newline)
# end use codecs
        
filename = './HICSS_authors.txt'

authors = []
authorcontents = []


# the co-author relationship matrix (2d array)
coauthornet = np.zeros([20000,20000],dtype=int)

"""
i = 0
with io.open(filename,'r',encoding='utf8') as f:
    for line in f:
        # split the authors by ','
        newline = line.split(",")
        # remove the last '\n' 
        newline.remove('\n')
        for name in newline:
            if name not in authors:
                authors.append(name)
                authorcontents.append(titles[i])
            else:    
                idx = authors.index(name)
                authorcontents[idx] = ' '.join([authorcontents[idx],titles[i]])
                        
        i = i + 1
"""

# use codecs
i = 0
f = codecs.open(filename,'r','utf-8')
for line in f:
    # split the authors by ','
    newline = line.split(",")
    # remove the last '\n' 
    newline.remove('\r\n')
    namelist = newline

    coauthornet = updatecoauthornetwork(coauthornet,authors,namelist)    
    
    for name in newline:
        if name not in authors:
            authors.append(name)
            authorcontents.append(titles[i])
        else:    
            idx = authors.index(name)
            authorcontents[idx] = ' '.join([authorcontents[idx],titles[i]])
                        
    i = i + 1
    print i
# end use codecs        
        
        
vectorizer = CountVectorizer(max_df=0.95, min_df=1,stop_words='english')

X = vectorizer.fit_transform(authorcontents)

analyze = vectorizer.build_analyzer()



Xarray = X.toarray()
hist = sum(Xarray)

#plt.plot(hist)

transformer = TfidfTransformer()

tfidf = transformer.fit_transform(Xarray)
tfidfarray = tfidf.toarray()

featurenames = vectorizer.get_feature_names()

# do the clustering

# number of clusters
n = 10 

km=KMeans(n_clusters=n, init='k-means++',n_init=50, verbose=1)

km.fit(tfidf)

# show the word cloud of the first cluster
#wordcloudit(' '.join(showcontents(km.labels_, 0, authorcontents)))


# gave the content of new author 

# authors.index('Jukka Huhtamäki'.decode('utf-8'))

authorIdx = authors.index('Jukka Huhtamäki'.decode('utf-8'))
content=[]

"""
if user is not in the author list:
otherwise we can use the tfidf feature directly
""" 

#if authorIdx < 0:
#    # let's test with a exist author, the author with index 3, in our list
#
#    content.append(authorcontents[authorIdx])
#    
#    # reuse the CountVectorizer vocabulary
#
#    reuseVect = CountVectorizer(vocabulary=vectorizer.vocabulary_)
#
#    # generate word count for new content based on exist vocabulary
#
#    contentvect = reuseVect.fit_transform(content)
#
#    contentvectarray = contentvect.toarray();
#
#    # append to the exist word vect array
#
#    #newXarray = np.concatenate((Xarray,contentvectarray.reshape((1,len(contentvectarray)))),axis=0)
#    newXarray = np.concatenate((Xarray,contentvectarray),axis=0)
#    # tfidf 
#    newtransformer = TfidfTransformer()
#
#    newtfidf = newtransformer.fit_transform(newXarray)
#
#    # feature tfidf array
#    featuretfidf = newtfidf[-1].toarray()[0]
#else:
featuretfidf = tfidfarray[authorIdx]

# compute the distance between the given author and all the cluster centers
dis = []

for i in km.cluster_centers_:
    dis.append(distance(featuretfidf,i))
    

sortdis= np.sort(dis)

#minclusterdis = min(dis)
#maxclusterdis = max(dis)

minclusterdis = sortdis[0]
#maxclusterdis = sortdis[-1]
#medclusterdis = sortdis[len(dis)/2]
maxclusterdis = sortdis[1]
medclusterdis = sortdis[2]

# this author belongs to the closest cluster
    
clusterid = dis.index(minclusterdis)


# farest cluster
farcluster = dis.index(maxclusterdis)

# median cluster
medcluster = dis.index(medclusterdis)

print '-----------------------'
print 'Author: ' 
print authors[authorIdx] 
print 'belongs to cluster: ' 
print clusterid
print '-----------------------'

# distance matrix from k-means
dismatrix = km.transform(tfidf)


# the range to find close authors around you based on the 
# distance to the closest cluster center

[mindis, maxdis, radius] = radiusofcluster(km.labels_, clusterid, dismatrix)

#r = max(maxdis-minclusterdis, minclusterdis-mindis)
#r = (maxdis-mindis)/2
r = (max(dis)-min(dis))/10*2

# all the authors in the same cluster
closeidx = np.where(((dismatrix[:,clusterid]>=(minclusterdis-r)) & (dismatrix[:,clusterid]<=(minclusterdis+r)))==True)[0]

#closeidx = np.where(km.labels_ == clusterid)
#closeidx = np.array(closeidx)
#closeidx = closeidx.flatten()


faridx = np.where(((dismatrix[:,farcluster]>=(maxclusterdis-r)) & (dismatrix[:,farcluster]<=(maxclusterdis+r)))==True)[0]


medidx = np.where(((dismatrix[:,medcluster]>=(medclusterdis-r)) & (dismatrix[:,medcluster]<=(medclusterdis+r)))==True)[0]


closeauthorsidx = np.nonzero(np.in1d(closeidx, faridx))[0]

closeidx1 = closeidx[closeauthorsidx]

closeauthorsidx1 = np.nonzero(np.in1d(closeidx1, medidx))[0]

closeauthors = closeidx1[closeauthorsidx1]

# compute the distance between the user and all the closeauthors

closeauthordis = []

for i in closeauthors:
    closeauthordis.append(distance(tfidfarray[authorIdx],tfidfarray[i]))

closeauthordis = np.array(closeauthordis)

closeauthors = closeauthors[closeauthordis.argsort()]


print 'recommended authors who has similar research interests: '

recommendauthor = []

for i in closeauthors:
    recommendauthor.append(authors[i])

for c in range(0,20):
    #print '{} : {}'.format(recommendauthor[c], coauthornet[closeauthors[0],:][closeauthors[c]])
    print recommendauthor[c]
# to visualize the result

# word cloud of the cluster
# wordcloudit(' '.join(showcontents(km.labels_, clusterid, authorcontents)))

# word cloud of each close author, closeauthors[0], closeauthors[1] ... 
# wordcloudit(authorcontents[closeauthors[0]])


# co-authorship

coauthornet = coauthornet[0:len(authors), 0:len(authors)];
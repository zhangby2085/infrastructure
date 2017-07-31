# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 16:38:22 2017

@author: secoder
"""
import io

import nltk
from nltk.tokenize import RegexpTokenizer


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.cluster import KMeans

from collections import OrderedDict

from sklearn.metrics import pairwise_distances

import numpy as np
import scipy

import json
import itertools
import codecs

import cPickle as pickle

import traceback

from skimage import filters

class recommendationsys:
    
    def __init__(self, f_titles, f_authors, f_years, f_booktitle, n, npaper, nyear):
        
        # define how much to pull close for 1st degree coauthorship connections 
        self.degree1p = 0.1
        # define how much to pull close for 2nd degree coauthorship connections 
        # based on degree1p, which is degree1p * degree2p for 2nd degree connections
        self.degree2p = 0.5
        
        # by default we will filter out those don't have publications in recent 10 years
        self.activityyear = 10

        self.debug = 0       
        self.nremd = 3
        
        #----------------------
        self.f_titles = f_titles
        self.f_authors = f_authors
        self.f_years = f_years
        self.f_booktitle = f_booktitle
        self.clusternum = n
        
        self.npaper = npaper
        self.nyear = nyear
        self.keywordthreshold = 10
        #----------------------        
        
        self.debugmsg('start init', 0)
        self.docluster()
        self.initNLTKConditionalFreqDist()
        
        self.filterN = len(self.authors)
        self.debugmsg('end init\n', 0)
        
        
    """
    """
    def debugmsg(self, msg, lvl):
        if self.debug <= lvl:
            print msg
    
    """
    """    
    def resentpublicationsidx(self,authoridx):
        #print 'start recentpublications\n'
        resentpub = []
        
        idx = self.authortitlesidx[authoridx]
    
        # sort by years
        years = [self.years[i] for i in idx]
        years = np.array(years)
        years = years.argsort()
        idx = np.array(idx)[years]
        idx = idx.tolist()
        idx.reverse()
        
        # if the most recent publication is before the 'nyears' 
        # remove this one from the list
        if (int(self.years[idx[0]]) < self.nyear) or (len(idx) < self.npaper):
            return resentpub
        # ----  
        
        for i in idx:
            authorsjson = []      
        
            for author in self.coathors[i]:
                authorsjson.append(OrderedDict([("name",author)]))
            resentpub.append(OrderedDict([("title",self.rawtitles[i]),("authors",authorsjson), ("year",self.years[i]),("publicationVenue",self.booktitle[i])]))

        #print 'end recentpublications\n'
        return resentpub
    
    
    """
    """
    def resentpublications(self,name):
        #print 'start recentpublications\n'
        resentpub = []

        if isinstance(name, unicode):
             #idx = self.authors.index(name)
             idx = self.authordict.get(name)
        else:
             #idx = self.authors.index(name.decode('utf-8'))  
             idx = self.authordict.get(name.decode('utf-8'))
        
        idx = self.authortitlesidx[idx]
    
        # sort by years
        years = [self.years[i] for i in idx]
        years = np.array(years)
        years = years.argsort()
        idx = np.array(idx)[years]
        idx = idx.tolist()
        idx.reverse()
        
        # if the most recent publication is before the 'nyears' 
        # remove this one from the list
        if (int(self.years[idx[0]]) < self.nyear) or (len(idx) < self.npaper):
            return resentpub
        # ----  
        
        for i in idx:
            authorsjson = []      
        
            for author in self.coathors[i]:
                authorsjson.append(OrderedDict([("name",author)]))
            resentpub.append(OrderedDict([("title",self.rawtitles[i]),("authors",authorsjson), ("year",self.years[i]),("publicationVenue",self.booktitle[i])]))

        #print 'end recentpublications\n'
        return resentpub
        
    def initNLTKConditionalFreqDist(self):
        self.debugmsg('start initNLTK CFD\n', 0)
        pairs=[]

#        for title in self.titles:
#            pairs = pairs + list(nltk.bigrams(title.split()))

        pairs = nltk.bigrams(self.allcorp)
    
        self.cfd = nltk.ConditionalFreqDist(pairs)
        self.debugmsg('end initNLTK CFD\n', 0)
    
    def keyword(self,name):
        #print 'start  keyword\n'
        if isinstance(name, unicode):
             #idx = self.authors.index(name)
             idx = self.authordict.get(name)
        else:
             #idx = self.authors.index(name.decode('utf-8'))  
             idx = self.authordict.get(name.decode('utf-8'))
             
#        content = self.authorcontents[idx].lower()
#        
#        # get the unique words from the content
#        content = set(content.split())
#        
#        i = []
#        for c in content:
#            count = self.vectorizer.vocabulary_.get(c, 0)   
#            i.append(count)
#            
#        i = np.array(i)
#        i = i.argsort()
#        content = np.array(list(content))
#        content = content[i]
#        content = content[-3:]
#        keywords = list(reversed(content))   
#        
        contentjson = []
#        for topic in keywords:
#            contentjson.append(OrderedDict([("topic", topic)]))
            
        # bigram keywords -------------
        content = self.authorcontents[idx].lower().split()
        finalkeywords = self.bigramkeywords(content)
#        #print 'start bigram\n'
#        
#        userpairs = list(nltk.bigrams(content))
#                
#       
#        # do the same on raw titles 
#        
#        keywordsraw=[]
#        for p in userpairs:
#            pairsdic=self.cfd[p[0]]
#            n=pairsdic[p[1]]
#            if n>=2:
#                keywordsraw.append((p,n))
#            
#        uniqkeywords=set(keywordsraw)
#        keywords=sorted(uniqkeywords, key=lambda keywords: keywords[1])
#        
#        finalkeywords=[]
#        for p in keywords:
#            #c=wn.synsets(p[0][1])[0].pos()
#            if (p[1]>=2):
#                finalkeywords.append((' '.join(p[0]),p[1],keywordsraw.count(p)))
#                
#        finalkeywords.reverse()
        
        for topic in finalkeywords:
            #print topic[0]
            contentjson.append(OrderedDict([("topic", topic[0])]))
        
        #print 'end bigram\n'
        #print 'end  keyword\n'
        return contentjson
    
    """
    """
    def keywordbyidx(self,idx):
             
        contentjson = []
            
        # bigram keywords -------------
        content = self.authorcontents[idx].lower().split()
        finalkeywords = self.bigramkeywords(content)
        
        for topic in finalkeywords:
            #print topic[0]
            contentjson.append(OrderedDict([("topic", topic[0])]))
        
        return contentjson

    
    """
    """    
    def bigramkeywords(self, text):
        #print 'start  bigramkeyword\n'
        # bigram keywords -------------
        #content = text.lower().split()
        content = text
        #print 'start bigram\n'
        
        userpairs = list(nltk.bigrams(content))
                
       
        # in case there is no valid keywords due to our requirement
        # the one with highest occurrence will be pick from the backup plan 
        keywordsbackup = []
        # the valid keywords
        keywords=[]
        for p in userpairs:
            pairsdic=self.cfd[p[0]]
            n=pairsdic[p[1]]
            if n>=self.keywordthreshold:
                keywords.append((p,n))
            keywordsbackup.append((p,n))

        finalkeywords=[]
        
        uniqkeywords=set(keywords)
        keywords=sorted(uniqkeywords, key=lambda keywords: keywords[1])
        for p in keywords:
            if (p[1]>=25) or (userpairs.count(p[0])>1):
                finalkeywords.append([' '.join(p[0]),p[1],userpairs.count(p[0])])
                
        finalkeywords.reverse()        
        
        
        
        if not finalkeywords:
            # found valid keywords
            uniqkeywords=set(keywordsbackup)
            keywordsbackup=sorted(uniqkeywords, key=lambda keywordsbackup: keywordsbackup[1])
            finalkeywords.append([' '.join(keywordsbackup[-1][0]), keywordsbackup[-1][1],userpairs.count(keywordsbackup[0])])
        else:        
            # deal with plural
            pluralidx = self.findpluralbigram(finalkeywords)
        
            self.removepluralbigram(finalkeywords,pluralidx)
        
        
        #print 'end  bigramkeyword\n'
        return finalkeywords
       
    """
    """
    def removepluralbigram(self, bigram, pluralidx):
        for i in pluralidx:
            delcount = 0
            for n in i[1:]:
                bigram[i[0]][1] = bigram[i[0]][1] + bigram[n-delcount][1]
                bigram.remove(bigram[n-delcount])
                delcount = delcount + 1
            
    
    """
    """
    def findpluralbigram(self, keywordsinfo):
        c = []
        for i in keywordsinfo:
            t = i[0].split()
            t1 = ''
            for n in t:
                if n[-1] == 's':
                    n = n[:-1]
                t1 = t1 + n

            c.append(t1)
            
        uniqbigram = list(set(c))
        pluralidx = []
        
        for i in uniqbigram:
            count = c.count(i)
            if count > 1:
                cc = []
                for n in range(len(c)):
                    if i == c[n]:
                        cc.append(n)
                pluralidx.append(cc)
         
        return pluralidx
    """
    """
    def mycoauthorsV2(self, name):
        if isinstance(name, unicode):
             #idx = self.authors.index(name)
             idx = self.authordict.get(name)
        else:
             #idx = self.authors.index(name.decode('utf-8'))  
             idx = self.authordict.get(name.decode('utf-8'))
             
        coauthorship = self.coauthornetV2[idx]
        uniqcoauthors = np.array(list(set(coauthorship)))
        coauthorcount = []
        for i in uniqcoauthors:
            coauthorcount.append(coauthorship.count(i))
            
        countidx = np.argsort(coauthorcount)
        # reverse it to descend order
        countidx = countidx[::-1]
        
        coauthorcount = np.array(coauthorcount)
        
        result = []
        for i in countidx:
            result.append(OrderedDict([("name",self.authors[uniqcoauthors[i]]),("cooperationCount",coauthorcount[i])]))
        return (result,list(uniqcoauthors[countidx]),list(coauthorcount[countidx]))
        
        
       
    """
    """
    def mycoauthorsV3(self, name):
        if isinstance(name, unicode):
             #idx = self.authors.index(name)
             idx = self.authordict.get(name)
        else:
             #idx = self.authors.index(name.decode('utf-8'))  
             idx = self.authordict.get(name.decode('utf-8'))
             
        coauthors = []
        for i in self.coauthorsidx:
            if idx in i:
                # remove itself
                t = i[:]
                t.remove(idx)
                coauthors.extend(t)
                
        coauthors = np.array(coauthors)
        unicoauthors, coauthorcount = np.unique(coauthors, return_counts=True)
        
        unicoauthors = unicoauthors[coauthorcount.argsort()]
        coauthorcount.sort()
        
        result = []
        for i in range(len(coauthorcount)):
            result.append(OrderedDict([("name",self.authors[unicoauthors[-(i+1)]]),("cooperationCount",coauthorcount[-(i+1)])]))
        return (result,list(unicoauthors[::-1]),list(coauthorcount[::-1]))

    """
    """
    def mycoauthorsV4(self, name):
        
        if isinstance(name, unicode):
             idx = self.authordict.get(name)
        else:
             idx = self.authordict.get(name.decode('utf-8'))
             
        coauthors = []
        for i in self.coauthorsidx:
            if idx in i:
                # remove itself
                t = i[:]
                t.remove(idx)
                coauthors.extend(t)
                
        coauthors = np.array(coauthors)
        unicoauthors, coauthorcount = np.unique(coauthors, return_counts=True)
        
        unicoauthors = unicoauthors[coauthorcount.argsort()]
        coauthorcount.sort()
        
        result = []
        for i in range(len(coauthorcount)):
            result.append(OrderedDict([("name",self.authors[unicoauthors[-(i+1)]]),("cooperationCount",coauthorcount[-(i+1)])]))
        
        
        return (result,list(unicoauthors[::-1]),list(coauthorcount[::-1]))
    
    """
    """
    def mycoauthorsV4byidx(self, idx):
        
        coauthors = []
        for i in self.coauthorsidx:
            if idx in i:
                # remove itself
                t = i[:]
                t.remove(idx)
                coauthors.extend(t)
                
        coauthors = np.array(coauthors)
        unicoauthors, coauthorcount = np.unique(coauthors, return_counts=True)
        
        unicoauthors = unicoauthors[coauthorcount.argsort()]
        coauthorcount.sort()
        
        result = []
        for i in range(len(coauthorcount)):
            result.append(OrderedDict([("name",self.authors[unicoauthors[-(i+1)]]),("cooperationCount",coauthorcount[-(i+1)])]))
        
        
        return (result,list(unicoauthors[::-1]),list(coauthorcount[::-1]))
        
     

    """
    radius of the cluster
    """
    def radiusofcluster(self, labels, nth, dismatrix):
        idx = np.where(labels == nth)[0]
    
        dis = dismatrix[idx,nth]
        
        self.mindis = min(dis)
        self.maxdis = max(dis)
        
        self.radius = self.maxdis
        
        
        
        # return [mindis, maxdis, radius]
            
    
    """
    show contents in the same cluster
    """
    def showcontents(self,labels, nth, allcontents):
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
    def digstring(self,s):
        for i in s:
            if i.isdigit():
                return True
        return False
        
    """
    compute the distance between two points a and b
    """
    def distance(self,a,b):

        if scipy.sparse.issparse(a):
            a = a.toarray()
            a = a[0]
            
        if scipy.sparse.issparse(b):
            b = b.toarray()
            b = b[0]
        
        a = np.array(a);
        b = np.array(b);
        return np.sqrt(sum(np.square(a - b)))
        
    """
    """
    def updatecoauthornetworkV2(self,net,authors,namelist):
        nameidx = []
        for name in namelist:
            nameidx.append(authors.index(name))
        
        for i in nameidx:
            tmpidx = nameidx[:]
            tmpidx.remove(i)
            # if net is empty
            if not net:
                net.append(tmpidx)
            else:
                if i>len(net)-1:
                    net.append(tmpidx)
                else:
                    net[i].extend(tmpidx)
        
    """
    """
    def docluster(self):
        tokenizer = RegexpTokenizer(r'\w+')

        #titledict={}
        self.rawtitles = []
        self.titles = []
        self.allcorp = []
        #newtitles = []
        sw = set(nltk.corpus.stopwords.words('english'))
        
        
        # filename = './CHI/CHI_titles.txt'
        self.debugmsg('start  titles \n', 0)
        f = codecs.open(self.f_titles,'r','utf-8')
        for line in f:   
            # remove the '.,\r\n' at the end
            line = line[:-4]
            self.rawtitles.append(line)
            line = line.lower()
            newline=tokenizer.tokenize(line)
            
            for corp in newline:
                self.allcorp.append(corp)
            
            # collect all the words except digtals and stopwords
            newline= ' '.join([w for w in newline if (w.lower() not in sw) & ~(self.digstring(w))])
            self.titles.append(newline)
        # end use codecs
                
        # filename = './CHI/CHI_authors.txt'
        self.authordict = {}
        self.authors = []
        self.authorcontents = []
        self.authorrawcontents = []
        self.authortitlesidx = []
        self.coathors = []
        self.coauthorsidx = []
                
        # the co-author relationship matrix (2d array)
        #self.coauthornet = np.zeros([num*3,num*3],dtype=int)
        self.coauthornetV2 = []
        
        # read years
        self.debugmsg('start  year \n', 0)
        self.years = []
        f = codecs.open(self.f_years,'r','utf-8')
        for line in f:
            # remove the '\r\n'
            line = line[:-2]
            self.years.append(line)
            
        # read conference 
        self.debugmsg('start  booktitle \n', 0)
        self.booktitle = []
        f = codecs.open(self.f_booktitle,'r','utf-8')
        for line in f:
            # remove the '\r\n'
            line = line[:-2]
            self.booktitle.append(line)
        
        # read authors
        self.debugmsg('start  authors \n', 0)
        i = 0
        m = 0
        f = codecs.open(self.f_authors,'r','utf-8')
        for line in f:
            # split the authors by ','
            newline = line.split(",")
            # remove the last '\n' 
            newline.remove('\r\n')
            namelist = newline
            self.coathors.append(namelist)           
            
            #self.coauthornet = self.updatecoauthornetwork(self.coauthornet,self.authors,namelist)    
            authoridx = []
            
            for name in newline:  
                
                # list version
#                if name not in self.authors:
#                    self.authors.append(name)
#                    self.authorcontents.append(self.titles[i])
#                    self.authorrawcontents.append(self.rawtitles[i])
#                    self.authortitlesidx.append([i])
#                    #idx = self.authors.index(name)
#                    #print 'idx: ' + str(idx) + ' m: ' + str(m)
#                    idx = m
#                    m = m + 1
#                else:    
#                    idx = self.authors.index(name)
#                    self.authortitlesidx[idx].append(i)
#                    self.authorcontents[idx] = ' '.join([self.authorcontents[idx],self.titles[i]])
#                    self.authorrawcontents[idx] = ' '.join([self.authorrawcontents[idx],self.rawtitles[i]])
#                authoridx.append(idx)
                # end list version 
                
                # dictonary version 
                idx = self.authordict.get(name)
                if idx:
                    self.authortitlesidx[idx].append(i)
                    #self.authorcontents[idx] = ' '.join([self.authorcontents[idx],self.titles[i]])
                    #self.authorrawcontents[idx] = ' '.join([self.authorrawcontents[idx],self.rawtitles[i]])
                    #self.authorcontents[idx].append(self.titles[i])
                    #self.authorrawcontents[idx].append(self.rawtitles[i])
                    self.authorcontents[idx] = self.authorcontents[idx] + ' ' + self.titles[i]
                    self.authorrawcontents[idx] = self.authorrawcontents[idx] + ' ' + self.rawtitles[i]
                else:
                    self.authors.append(name)
                    #self.authordict.update({name:m})
                    self.authordict[name] = m
                    #self.authorcontents.append(self.titles[i])
                    #self.authorrawcontents.append(self.rawtitles[i])
                    #self.authorcontents.append([self.titles[i]])
                    #self.authorrawcontents.append([self.rawtitles[i]])
                    self.authorcontents.append(self.titles[i])
                    self.authorrawcontents.append(self.rawtitles[i])
                    
                    self.authortitlesidx.append([i])
                    #idx = self.authors.index(name)
                    #print 'idx: ' + str(idx) + ' m: ' + str(m)
                    idx = m
                    m = m + 1
                authoridx.append(idx)
                # end  dict version
                
            self.coauthorsidx.append(authoridx)
            #self.updatecoauthornetworkV2(self.coauthornetV2,self.authors,namelist)
            i = i + 1
            #print i
        # end use codecs
        #self.coauthornet = self.coauthornet[0:len(self.authors), 0:len(self.authors)];
           
        #return
                
        self.vectorizer = CountVectorizer(max_df=0.95, min_df=1,stop_words='english')
        
        X = self.vectorizer.fit_transform(self.authorcontents)
        
        #Xarray = X.toarray()
        Xarray = X
        
        #plt.plot(hist)
        
        transformer = TfidfTransformer()
        
        self.tfidf = transformer.fit_transform(Xarray)
        #self.tfidfarray = self.tfidf.toarray()
        self.tfidfarray = self.tfidf
        
        self.featurenames = self.vectorizer.get_feature_names()
        
        # do the clustering
        
        # number of clusters
        # n = 10 
        
        #self.km=KMeans(n_clusters=self.clusternum, init='k-means++',n_init=50, verbose=1)
        
        #self.km.fit(self.tfidf)
        
    
    """
    """
    def recommendationV2(self, name, n):
        self.debugmsg('find the idx', 0)
        if isinstance(name, unicode):
             #idx = self.authors.index(name)
             authorIdx = self.authordict.get(name)
        else:
             #idx = self.authors.index(name.decode('utf-8'))  
             authorIdx = self.authordict.get(name.decode('utf-8'))
             name = name.decode('utf-8')
        #content=[]
    
        self.myidx = authorIdx  
        self.debugmsg('get the feature vector', 0)
        featuretfidf = self.tfidfarray[authorIdx]
        
        self.debugmsg('start distance computing \n', 0)
        (self.closeauthors, self.closeauthordis) = self.nNNlinesearch(self.tfidfarray,featuretfidf,0)
        self.debugmsg('end distance computing \n', 0)
        
        self.recommendauthor = []
        for i in self.closeauthors:
            self.recommendauthor.append(self.authors[i]) 
            
        # remove userself 
        try:
            selfidx = self.recommendauthor.index(name)
        except ValueError:
            selfidx = -1
            
        self.debugmsg('selfidx is ' + str(selfidx), 0)
        
        if selfidx != -1:
            self.closeauthordis = np.delete(self.closeauthordis,selfidx)
            self.recommendauthor.remove(name) 
    
    
        """
            print the top 20 unfilted recommentdauthor
        """
        i=0
        for c in self.recommendauthor:
            #print '{} : {}'.format(recommendauthor[c], coauthornet[closeauthors[0],:][closeauthors[c]])
            print c
            i=i+1;
            if i>20:
                break;
                
        self.debugmsg('filter recommendtion \n', 0)
        tmpremd = self.filteredrecommendations(self.filterN)
        
        self.debugmsg('threshold recommendtion \n', 0)
        newrecommendations = self.thresholdrecommendations(tmpremd,n)
    
        self.result=OrderedDict([("name",name),("recommendations",newrecommendations)])        
        self.debugmsg('end recommendationV2 \n', 0)
        return self.result    
    
    
    
    """    
    """
    def recommendationV3(self, name, n):
        self.nremd = n
        self.debugmsg('Will generate recommendations in 3 groups and ' + str(n) + ' for each group', 1)
        self.debugmsg('find the idx', 0)
        if isinstance(name, unicode):
             #idx = self.authors.index(name)
             authorIdx = self.authordict.get(name)
        else:
             #idx = self.authors.index(name.decode('utf-8'))  
             authorIdx = self.authordict.get(name.decode('utf-8'))
             name = name.decode('utf-8')
        #content=[]
    
        self.myidx = authorIdx  
        self.debugmsg('get the feature vector', 0)
        featuretfidf = self.tfidfarray[authorIdx]
        
        self.debugmsg('start distance computing \n', 0)
        (self.closeauthors, self.closeauthordis) = self.nNNlinesearch(self.tfidfarray,featuretfidf,0)
        self.debugmsg('end distance computing \n', 0)

        # here we can define the range to apply the otsu for recommendations
        # for example self.closeauthordis[0:1000] or all them
        self.debugmsg('start otsuifilter\n', 0)
        splitidx = self.otsufilter(self.closeauthordis)   
        self.debugmsg('end otsufilter\n', 0)                    
        
        # splitidx contains the first index of three groups, close, medium, far
        # now generate three recommendations in each group
        recommendations = []
        
        # save the valid remdidx
        remdidx = []
        for i in splitidx:
            n = 0
            backwardcount = 1
            while n != self.nremd:
                if self.closeauthors[i] != self.myidx:
                    # skip myself go to next one
                    remdinfo = self.getremdinfo(i)
                    if remdinfo and ~remdidx.count(i):
                        #print remdinfo
                        recommendations.append(remdinfo)
                        n = n + 1
                        remdidx.append(i)
                        #self.debugmsg(str(n) + ' ' + str(i), 0)
                        
                i = i + 1
                
                # didn't find required number of valid remd untill the end
                # start backwards search
                if (i == len(self.closeauthordis)) or (backwardcount > 1):
                    if backwardcount == 1:
                        backwardstart = i - self.nremd
                    i = backwardstart - backwardcount
                    backwardcount = backwardcount + 1
                    #self.debugmsg('search backward ' + str(i), 0)
        

    
    
        self.result=OrderedDict([("name",name),("recommendations",recommendations)])        
        self.debugmsg('end recommendationV3 \n', 0)
        return self.result    
      

  
    """
        find n nearset neighbors of point p in given space using linear search
        if n == 0, sort all the points in space
    """
    def nNNlinesearch(self, space, p, n):
        closeauthordis = []
            
#        for i in space:
#            closeauthordis.append(self.distance(p,i))
            
        #closeauthordis = pairwise_distances(space, p)
        closeauthordis = pairwise_distances(space, p, metric='cosine')
        closeauthordis = closeauthordis.flatten()
        #closeauthordis = np.array(closeauthordis)
            
        closeauthors = closeauthordis.argsort()

        if n > 0 :
            closeauthors = closeauthors[0:n]
            closeauthordis = closeauthordis[0:n]
      
        closeauthordis.sort()
        
        return (closeauthors, closeauthordis)
        
        
            
    

    """
        find n nearset neighbors of point p in given space using D positioning
    """
    def nNNPositioning(self, space, p, n):
        # get the dimention of the space
        # compute the distance between the given author and all the cluster centers
        dis = []
        
        for i in self.km.cluster_centers_:
            dis.append(self.distance(p,i))
            
        
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
        print self.authors[self.myidx] 
        print 'belongs to cluster: ' 
        print clusterid
        print '-----------------------'
        
        # distance matrix from k-means
        dismatrix = self.km.transform(self.tfidf)
        
        
        # the range to find close authors around you based on the 
        # distance to the closest cluster center
        
        #[mindis, maxdis, radius] = self.radiusofcluster(self.km.labels_, clusterid, dismatrix)
        
        l = 0
        p = 1    

        
        while l < 200:
       # while l < n*5:
            #r = max(maxdis-minclusterdis, minclusterdis-mindis)
            #r = (maxdis-mindis)/2
            r = (max(dis)-min(dis))/10*p
            #r = 0.5*p
        
            closeauthors = []
            closeauthordis = []            
            
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
                       
            for i in closeauthors:
                closeauthordis.append(self.distance(p,space[i]))
            
            closeauthordis = np.array(closeauthordis)
            
            closeauthordis = closeauthordis[closeauthordis<=r]
            
            closeauthors = closeauthors[closeauthordis.argsort()]
            l = len(closeauthors)
            p = p+1
            print 'l {}, r {}, p {}'.format(l,r,p)
        
        
        self.closeauthordis.sort()
        
        return (closeauthors, closeauthordis)
    
    """
    """
    def recommendation(self, name, n):
        # authors.index('Jukka Huhtamäki'.decode('utf-8'))
    
        if isinstance(name, unicode):
            authorIdx = self.authors.index(name)
        else:
            name = name.decode('utf-8')
            authorIdx = self.authors.index(name)
        #content=[]
    
        self.myidx = authorIdx    
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
        #
        featuretfidf = self.tfidfarray[authorIdx]
        #featuretfidf = self.tfidf[authorIdx]
    
#        # compute the distance between the given author and all the cluster centers
#        dis = []
#        
#        for i in self.km.cluster_centers_:
#            dis.append(self.distance(featuretfidf,i))
#            
#        
#        sortdis= np.sort(dis)
#        
#        #minclusterdis = min(dis)
#        #maxclusterdis = max(dis)
#        
#        minclusterdis = sortdis[0]
#        #maxclusterdis = sortdis[-1]
#        #medclusterdis = sortdis[len(dis)/2]
#        maxclusterdis = sortdis[1]
#        medclusterdis = sortdis[2]
#        
#        # this author belongs to the closest cluster
#            
#        clusterid = dis.index(minclusterdis)
#        
#        
#        # farest cluster
#        farcluster = dis.index(maxclusterdis)
#        
#        # median cluster
#        medcluster = dis.index(medclusterdis)
#        
#        print '-----------------------'
#        print 'Author: ' 
#        print self.authors[authorIdx] 
#        print 'belongs to cluster: ' 
#        print clusterid
#        print '-----------------------'
#        
#        # distance matrix from k-means
#        dismatrix = self.km.transform(self.tfidf)
#        
#        
#        # the range to find close authors around you based on the 
#        # distance to the closest cluster center
#        
#        #[mindis, maxdis, radius] = self.radiusofcluster(self.km.labels_, clusterid, dismatrix)
#        
#        l = 0
#        p = 1    
#        
#        while l < 200:
#       # while l < n*5:
#            #r = max(maxdis-minclusterdis, minclusterdis-mindis)
#            #r = (maxdis-mindis)/2
#            r = (max(dis)-min(dis))/10*p
#            #r = 0.5*p
#            
#            # all the authors in the same cluster
#            closeidx = np.where(((dismatrix[:,clusterid]>=(minclusterdis-r)) & (dismatrix[:,clusterid]<=(minclusterdis+r)))==True)[0]
#            
#            #closeidx = np.where(km.labels_ == clusterid)
#            #closeidx = np.array(closeidx)
#            #closeidx = closeidx.flatten()
#            
#            
#            faridx = np.where(((dismatrix[:,farcluster]>=(maxclusterdis-r)) & (dismatrix[:,farcluster]<=(maxclusterdis+r)))==True)[0]
#            
#            
#            medidx = np.where(((dismatrix[:,medcluster]>=(medclusterdis-r)) & (dismatrix[:,medcluster]<=(medclusterdis+r)))==True)[0]
#            
#            
#            closeauthorsidx = np.nonzero(np.in1d(closeidx, faridx))[0]
#            
#            closeidx1 = closeidx[closeauthorsidx]
#            
#            closeauthorsidx1 = np.nonzero(np.in1d(closeidx1, medidx))[0]
#            
#            self.closeauthors = closeidx1[closeauthorsidx1]
#            
#            # compute the distance between the user and all the closeauthors
#            
#            self.closeauthordis = []
#            
#            for i in self.closeauthors:
#                self.closeauthordis.append(self.distance(self.tfidfarray[authorIdx],self.tfidfarray[i]))
#            
#            self.closeauthordis = np.array(self.closeauthordis)
#            
#            self.closeauthordis = self.closeauthordis[self.closeauthordis<=r]
#            
#            self.closeauthors = self.closeauthors[self.closeauthordis.argsort()]
#            l = len(self.closeauthors)
#            p = p+1
#            print 'l {}, r {}, p {}'.format(l,r,p)
#        
#        
#        self.closeauthordis.sort()
    
    
        (self.closeauthors, self.closeauthordis) = self.nNNPositioni(self.tfidfarray, featuretfidf, 0)
        #print 'After {} got {},  recommended authors who has similar research interests: '.format(p,l)
        
        
        self.recommendauthor = []
        for i in self.closeauthors:
            self.recommendauthor.append(self.authors[i])

        # remove userself 
        try:
            selfidx = self.recommendauthor.index(name)
        except ValueError:
            selfidx = -1
            
        print "selfidx is %d" % selfidx
        
        if selfidx != -1:
            self.closeauthordis = np.delete(self.closeauthordis,selfidx)
            self.recommendauthor.remove(name)        
        
        """
            print the top unfilted recommentdauthor
        """
        i=0
        for c in self.recommendauthor:
            #print '{} : {}'.format(recommendauthor[c], coauthornet[closeauthors[0],:][closeauthors[c]])
            print c
            i=i+1;
            if i>20:
                break;
                
        tmpremd = self.filteredrecommendations(self.filterN)
        
        newrecommendations = self.thresholdrecommendations(tmpremd,n)
    
        result=OrderedDict([("name",name),("recommendations",newrecommendations)])        
        
        return result

    """
        split the distance in to 3 groups using otsu filtering
        return the first index of each group
    """
    def otsufilter(self, tdis):
        trd = np.zeros(3, int)
        
        #tdis = self.filteredcloseauthordis()
        t1 = filters.threshold_otsu(tdis)
        t2 = filters.threshold_otsu(tdis[tdis>t1])
         
        # the first index of each group
#        trd[1] = len(tdis[tdis<t1])
#        trd[2] = len(tdis) - len(tdis[tdis>t2])
        
        # get the medium 3 in the medium group
        # get the last 3 in the far group
        trd[1] = int((len(tdis[tdis<t2]) - len(tdis[tdis<t1]))/2)-1
        trd[2] = len(tdis) - 3         
        
        return trd

    """
        extract the detail inforamtion of the recommendation by its indx in
        the closeauthors
        ignor those unqualified ones which has few papers or not active 
        recently, and also remove my co-authors
    """
    def getremdinfo(self, clsidx):
        # get the author index from closeauthors
        remdidx = self.closeauthors[clsidx]
        
        recentpub = self.resentpublicationsidx(remdidx)
        
        if recentpub:
            name = self.authors[remdidx]
            [coauthors, idx, c] = self.mycoauthorsV4byidx(remdidx)
            
            if idx.count(self.myidx):
                # remove the coauthor
                return []
            
            researchtopic = self.keywordbyidx(remdidx)
            return OrderedDict([("name",name), ("relevancy",self.closeauthordis[clsidx]),("coAuthors",coauthors),("researchTopics",researchtopic), ("recentPublications",recentpub)])
        else:
            return []

    """
    """
    def updatedistance(self):
        # 1st degree connection in coauthorship
        deg1con=self.coauthornet[self.myidx,self.closeauthors]
        deg1conidx = np.where(deg1con>0)[0]
        #deg1con = deg1con[deg1con>0]
        
        # 2nd degree connection in coauthorship
        deg2conidx = np.where(deg1con==0)[0]
        deg2con = np.zeros(deg2conidx.size)
        
        for i in self.closeauthors[deg1conidx]:
            deg2con = deg2con + self.coauthornet[i,self.closeauthors[deg2conidx]]
            
        deg1con = deg1con[deg1con>0]
        
        deg1con = deg1con/max(deg1con)
        return (deg1conidx, deg1con,deg2conidx,deg2con)
        
    """
        return the top N recommendations:
        recommendations, coauthors, researchtopics, recentpub(at least 3 and no 
        morethan  5 years) 
    """
    def filteredrecommendations(self, n):
        
        recommendations = []
        self.filteridx = []
        self.filteredauthors = []
        
        i = 0
        for name in self.recommendauthor:
            #coauthors = []
            #researchtopic = []
            #recentpub = []
            #coauthorsjson = []
            #[coauthors, idx, c] = self.mycoauthors(name)
            #[coauthors, idx, c] = self.mycoauthorsV2(name)
            #[coauthors, idx, c] = self.mycoauthorsV3(name)
            [coauthors, idx, c] = self.mycoauthorsV4(name)

            # remove the coauthors  
            if idx.count(self.myidx):
                i = i+1
                continue
            
            recentpub = self.resentpublications(name)

            # check if the recentpub is empty which is not active anymore
            if not recentpub:
                i = i+1
                continue
            # --    

            self.filteredauthors.append(name)            
            
            # take too much time skip in test
            # researchtopic = self.keyword(name)
            researchtopic = []
            researchtopic.append(OrderedDict([("topic", "TBD")]))
            
    
            #recommendations.append({'name':name, 'coAuthors':coauthors, 'researchTopcs':researchtopic, 'recentPublications':recentpub} )
            recommendations.append(OrderedDict([("name",name), ("relevancy",self.closeauthordis[i]),("coAuthors",coauthors),("researchTopics",researchtopic), ("recentPublications",recentpub)])) 
            #result={'name':user, 'recommendations':recommendations};
            
            # save the picked idx
            self.filteridx.append(i)    
            i = i+1
            
            # only need top n recommendations
            
            if len(self.filteridx) == n:
                break
        
        return recommendations
        
        
    """
    """
    def thresholdrecommendations(self, remds,n):
        
        thredremd = []
        self.trd = np.zeros(3)
        
        tdis = self.filteredcloseauthordis()
        t1 = filters.threshold_otsu(tdis)
        t2 = filters.threshold_otsu(tdis[tdis>t1])
         
        # get the top 3 in each group
        self.trd[1] = len(tdis[tdis<t1])
        self.trd[2] = len(tdis) - len(tdis[tdis>t2])
        
        # get the top 3 in first group, median 3 in second group, 
        # last 3 in third group
#        self.trd[1] = int((len(tdis[tdis<t2]) - len(tdis[tdis<t1]))/2)-1
#        self.trd[2] = len(tdis) - 3
         
         
        for i in range(3):
            for j in range(int(n/3)):
                k = int(self.trd[i]+j)
                name = remds[k]['name']
                researchtopic = self.keyword(name)
                remds[k]['researchTopics'] = researchtopic
                thredremd.append(remds[k])
                
        return thredremd
            

    
    """
    """
    def filteredcloseauthordis(self):
        return self.closeauthordis[self.filteridx]
    
    """
    """
    def save_object(self, filename):
        # clear the user data
        self.filteredauthors=[]
        self.filteridx=[]       
        self.recommendauthor = []
        
        try:
            with open(filename, 'wb') as output:
                pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
        except pickle.UnpicklingError as e:
            # normal, somewhat expected
            print('UnpicklingError')
        except (AttributeError,  EOFError, ImportError, IndexError) as e:
            # secondary errors
            print(traceback.format_exc(e))
        
        except Exception as e:
            # everything else, possibly fatal
            print(traceback.format_exc(e))        

        return
        
    """
    """
    def save_json(self,filename):  
        with io.open(filename+'.json','w',encoding="utf-8") as outfile:
            outfile.write((json.dumps((self.result), ensure_ascii=False)))
   
        
"""
  Start 
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 16:38:22 2017

@author: secoder
"""
import io
import random
import nltk
from nltk.tokenize import RegexpTokenizer


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


from collections import OrderedDict
from collections import Counter

from sklearn.metrics import pairwise_distances

import numpy as np
import scipy

import json
import codecs

from dateutil import parser
import time
import datetime

import operator

#import cPickle as pickle
#
#import traceback

from skimage import filters

import unicodedata as ud

from config import project_name


class recommendationsys:
    
    def __init__(self, nyear):
        
        # by default we will filter out those don't have publications in recent 10 years
        self.activityyear = 10

        self.debug = 0       
        self.nremd = 3
        
        #----------------------
        PROJECT_DIRECTORY = 'output/project/' + project_name

        self.f_titles = PROJECT_DIRECTORY + '/cleantitles_target.txt'
        self.f_authors = PROJECT_DIRECTORY + '/authors_target.txt'
        self.f_years = PROJECT_DIRECTORY + '/years_target.txt'
        self.f_booktitle = PROJECT_DIRECTORY + '/venues_target.txt'
        self.f_mentionnetwork = PROJECT_DIRECTORY + '/mention_network.txt'
        self.f_perorglabel = PROJECT_DIRECTORY + '/per_org.txt'
        self.f_authors_id = PROJECT_DIRECTORY + '/authors_id_target.txt'

        
        self.npaper = 10
        self.nyear = time.mktime(parser.parse(str(nyear)).timetuple())
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
            print(msg)
    
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
            
            date = datetime.datetime.fromtimestamp(self.years[i]).strftime("%Y-%m-%d %H:%M:%S")
            resentpub.append(OrderedDict([("title",self.rawtitles[i]),("authors",authorsjson), ("year",date),("publicationVenue",self.booktitle[i])]))

        #print 'end recentpublications\n'
        return resentpub
    
    
    """
    """
    def resentpublications(self,name):
        #print 'start recentpublications\n'
        resentpub = []

        #if isinstance(name, unicode):  for python 2.7
        if isinstance(name, str):
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
            
            date = datetime.datetime.fromtimestamp(self.years[i]).strftime("%Y-%m-%d %H:%M:%S")
            resentpub.append(OrderedDict([("title",self.rawtitles[i]),("authors",authorsjson), ("year",date),("publicationVenue",self.booktitle[i])]))

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
        if isinstance(name, str):
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
        # if pluralidx is emtpy, just return
        if not pluralidx:
            print('empty')
            return 
        
        delcount = 0
        pren = 0        

        for i in pluralidx:
            #delcount = 0
            for n in i[1:]:
                if n > pren:
                    n = n - delcount

                bigram[i[0]][1] = bigram[i[0]][1] + bigram[n][1]
                bigram.remove(bigram[n])
                delcount = delcount + 1
                pren = n
            
    
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
        if isinstance(name, str):
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
        if isinstance(name, str):
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
        
        if isinstance(name, str):
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
    """
    def mycoauthorsV4bymentionlist(self, name):
        
        if name in self.mentionnetwork.keys():
            mentiondict = self.mentionnetwork[name]
        else:
            mentiondict ={'None':0}
            
        
        result = []
        # sort by mention counts
        sorted_mentiondict = sorted(mentiondict.items(), key=operator.itemgetter(1), reverse=True)
        
        for i in sorted_mentiondict:
            result.append(OrderedDict([("name",i[0]),("cooperationCount",i[1])]))
        
        return result
    """
    """
    def mycoauthorsbyyear(self, idx, year):
        
        years = np.array(self.years)

        yearidx = np.where(years <= year)[0]
        coauthorsidx = [ self.coauthorsidx[i] for i in yearidx]
        
        coauthors = []
        for i in coauthorsidx:
            if idx in i:
                # remove itself
                t = i[:]
                t.remove(idx)
                coauthors.extend(t)
                
        coauthors = np.array(coauthors)
        unicoauthors, coauthorcount = np.unique(coauthors, return_counts=True)
        
        unicoauthors = unicoauthors[coauthorcount.argsort()]
        coauthorcount.sort()
    
        return (list(unicoauthors[::-1]),list(coauthorcount[::-1]))
        
    """
        find the new coauthors for a user in current year against previous year
        example: mynewcoauthors(23, 2014, 2015) will returen the new coauthors
        in 2015 regarding the year 2014 for user 23. 23 is the index of a user
    """
    def mynewcoauthors(self, userIdx, yearPre, yearCur):
        coauthornetPre, cp = self.mycoauthorsbyyear(userIdx, yearPre)

        coauthornetCur, cc = self.mycoauthorsbyyear(userIdx, yearCur)

        newCoauthors = np.setdiff1d(coauthornetCur, coauthornetPre)
        
        return newCoauthors

    """
        Call the weakties after mynewcoauthors() to find the common nodes 
        between a user and his/her coming new coauthors in the year before
        their coauthorship
    """
    def weakties(self, userX, userY, year):
        
        coauthornetX, cx = self.mycoauthorsbyyear(userX, year)
        
        # if userX and userY already have a strong ties, just return []
        if userY in coauthornetX:
            return ([], [], [])
        
        coauthornetY, cy = self.mycoauthorsbyyear(userY, year)
        
        # find the common nodes 
        weaktienodes = list(set(coauthornetX).intersection(coauthornetY))
        
        nodescountX = []
        nodescountY = []
        
        if weaktienodes:
            for i in weaktienodes:
                nodescountX.append(cx[coauthornetX.index(i)])
                nodescountY.append(cy[coauthornetY.index(i)])
            
        
        return (weaktienodes, nodescountX, nodescountY)
    
    """
        2nd hoop connection
    """
    def secondhoopties(self, userX, userY, year):
        result = []
        coauthors1, count1 = self.mycoauthorsbyyear(userX, 2016)

        for i in coauthors1:
            coauthors2, count2 = self.mycoauthorsbyyear(i, 2016)
            for n in coauthors2:
                coauthors3, count3 = self.mycoauthorsbyyear(n, 2016)
                if userY in coauthors3:
                    result.append([[i,n],[count1[coauthors1.index(i)],count2[coauthors2.index(n)], count3[coauthors3.index(userY)]]])


    """
        Get all the content(paper titles) of the userIdx before 
        the 'year'(include the year) 
    """
    def getcontentbyyear(self, userIdx, year):
        titleIdx = self.authortitlesidx[userIdx]

        titleIdx = np.array(titleIdx)

        years = [self.years[i] for i in titleIdx]

        years = np.array(years)
        
        # sort the years and put the latest year first
        # then the content will also be sorted by recent paper first
        years.sort()
        years = years[::-1]

        yearIdx = np.where(years<=year)[0]
    
        content = [self.titles[i] for i in titleIdx[yearIdx]]
        
        return content

    """
        return the most frequent participated venue of a user
    """
    def getVenue(self, userIdx):
        venues = self.authorbooktitleidx[userIdx]
        c = Counter(venues)
        frqvenues = c.most_common()
        
        return frqvenues[0][0]

    """
        only consider the recent 10 papers
    """
    def contentsimilarity(self, userX, userY, year):
        contentX = self.getcontentbyyear(userX, year)
        if not contentX:
            return -1
        contentX = contentX[0:10]
        
        contentY = self.getcontentbyyear(userY, year)
        if not contentY:
            return -1
        contentY = contentY[0:10]
        
        # build the corpus of all the content
        contents = []
        
        
        for i in contentX:
            contents.extend(i.split(' '))
        
        lenx = len(contents)
        
        for i in contentY:
            contents.extend(i.split(' '))
        
        #  normalize the different forms of words 
        stemmer = nltk.stem.PorterStemmer()
        stems = [stemmer.stem(t) for t in contents]           
        
        # reconstruct content for userX and userY use the normalized words
        newcontentX = stems[0:lenx]
        newcontentY = stems[lenx:]


        
        vectorizer = CountVectorizer()
        v = vectorizer.fit_transform([' '.join(newcontentX), ' '.join(newcontentY)])
        
        cosinesimilarity = pairwise_distances(v[0], v[1], metric='cosine')[0][0]
        
        return cosinesimilarity

    """
        network similarity
    """
    def networksimilarity(self, userX, userY, year):
        
        # first calculate FG(userX) according to paper
        # User similarities on social networks
        coauthors, c = self.mycoauthorsbyyear(userX, year)
        
        edgesFG = len(coauthors)
    
        n = 0
        for i in coauthors:
            subcoauthors, c = self.mycoauthorsbyyear(i, year)
            con = list(set(subcoauthors).intersection(coauthors[n:]))
            edgesFG = edgesFG + len(con)
            n = n + 1
            
        # second, calculate MFG(userX, userY)
        weakties, cx, cy = self.weakties(userX, userY, year)
        
        edgesMFG = 2 * len(weakties)
        
        n = 0
        for i in weakties:
            subcoauthors, c = self.mycoauthorsbyyear(i, year)
            con = list(set(subcoauthors).intersection(weakties[n:]))
            edgesMFG = edgesMFG + len(con)
            n = n + 1
            
        # last calculate the network similarity
        
        if edgesFG * edgesMFG:
            ns = np.log(edgesMFG)/np.log(2 * edgesFG)
        else:
            ns = -1
            
        return (ns, edgesFG, edgesMFG, cx, cy)

    """
        text processing, normalize the words to their prototype, such as 
        plural form, progressive, etc
    """
    def textnormalizing(self, text):
        #l = len(text)
        c = 0
        for i in text:
            # network - networks
            if i[-1] == 's':
                ii = i[:-1]
                if ii in text:
                    text[c] = ii
                    c = c + 1
                    continue
                
            # bus - buses
            if i[-2:] == 'es':
                ii = i[:-2]
                if ii in text:
                    text[c] = ii
                    c = c + 1
                    continue
                
            #  study - studies 
            if i[-3:] == 'ies':
                ii = i[:-3] + 'y'
                if ii in text:
                    text[c] = ii
                    c = c + 1
                    continue
            
            # network - networking
            # get - getting
            # explore - exploring 
            if i[-3:] == 'ing':
                ii = i[:-3]
                if ii in text:
                    text[c] = ii
                    c = c + 1
                    continue
                
                ii = i[:-4]
                if ii in text:
                    text[c] = ii
                    c = c + 1
                    continue
                
                ii = i[:-3] + 'e'
                if ii in text:
                    text[c] = c + 1
                    continue
                
            c = c + 1
            
        return text
                
    """
    """

    
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
        load the person or organization label
    """
    def per_org_label(self):
        f = codecs.open(self.f_perorglabel,'r','utf-8')
        labels = {}
        for line in f:
            items = line.split()
            labels[items[0]] = items[1] 
        f.close()
        self.labels = labels

    """
    """
    def mention_network(self):
        f = codecs.open(self.f_mentionnetwork,'r','utf-8')
        source=''
        network = {}
        for line in f:
            items = line.split('"')
            if source == '':
                source = items[0]
                target = {}
		
            if source == items[0]:
                target[items[1]] = int(items[2])
            else:
                network[items[0]] = target
                source = items[0]
                target = {}
                
        f.close()
        return network
    
    
    """
    """
    def docluster(self):
        tokenizer = RegexpTokenizer(r'\w+')


        self.rawtitles = []
        self.titles = []
        self.allcorp = []

        sw = set(nltk.corpus.stopwords.words('english'))
        
        
        self.debugmsg('start  titles \n', 0)
        f = codecs.open(self.f_titles,'r','utf-8')
        for line in f:   
            # remove the '\n' at the end
            if line[-1] == '\n':
                line = line[:-1]
            self.rawtitles.append(line)
            line = line.lower()
            tokenlist = tokenizer.tokenize(line)
            
            self.allcorp += tokenlist
            #for corp in newline:
            #    self.allcorp.append(corp)
            
            # collect all the words except digtals and stopwords
            tokenlist = ' '.join([w for w in tokenlist if (w.lower() not in sw) & ~(self.digstring(w))])
            self.titles.append(tokenlist)
        f.close()
        # end use codecs
                
        # filename = './CHI/CHI_authors.txt'
        self.authordict = {}
        self.authors = []
        self.authorcontents = []
        self.authorrawcontents = []
        self.authortitlesidx = []
        self.authorbooktitleidx = []
        self.coathors = []
        self.coauthorsidx = [] # undirect link, etc, dblp coauthorship network
        self.mentionnetwork = {} # direct link, etc,tweet mention network
        self.id_name = {}
                

        self.coauthornetV2 = []
        
        # readin the mention network
        self.mentionnetwork = self.mention_network()
        
        # read years
        self.debugmsg('start  year \n', 0)
        self.years = []
        
        f = codecs.open(self.f_years,'r','utf-8')
        for line in f:
            # remive \n
            if line[-1] == '\n':
                line = line[:-1]
            if line == '':
                line = 0
            #line = line.split()
            #year = line[-1]
            timestamp = time.mktime(parser.parse(line).timetuple())
            self.years.append(int(timestamp))
        f.close()
        
        # read conference 
        self.debugmsg('start  booktitle \n', 0)
        self.booktitle = []
        
        f = codecs.open(self.f_booktitle,'r','utf-8')
        for line in f:
            # remove the \n at the end
            line = line[:-1]
            self.booktitle.append(line)
        f.close()
        
        # read authors
        self.debugmsg('start  authors \n', 0)
        i = 0
        m = 0
        f = codecs.open(self.f_authors,'r','utf-8')

        for line in f:
            # remove the last '\n' 
            line = line[:-1]
            # split the authors by ','
            newline = line.split(",")
            namelist = newline
            self.coathors.append(namelist)           
            
 
            authoridx = []
            
            for name in newline:  
                
                # dictonary version 
                idx = self.authordict.get(name)
                if idx is not None:
                    self.authortitlesidx[idx].append(i)
                    self.authorbooktitleidx[idx].append(i)

                    self.authorcontents[idx] = self.authorcontents[idx] + ' ' + self.titles[i]
                    self.authorrawcontents[idx] = self.authorrawcontents[idx] + ' ' + self.rawtitles[i]
                else:
                    self.authors.append(name)

                    self.authordict[name] = m

                    self.authorcontents.append(self.titles[i])
                    self.authorrawcontents.append(self.rawtitles[i])
                    
                    self.authortitlesidx.append([i])
                    self.authorbooktitleidx.append([i])

                    idx = m
                    m = m + 1
                authoridx.append(idx)
                # end  dict version
            
            self.coauthorsidx.append(authoridx)
            i = i + 1

        f.close()

        
        f = codecs.open(self.f_authors_id,'r','utf-8')
        i = 0
        preline = ''
        for line in f:
            if preline != line:
                #print(i)
                #print('preline: {}, line: {}'.format(preline, line))
                if line[-1] == '\n':
                    newline = line[:-1]
                self.id_name[self.authors[i]] = newline
                preline = line
                i = i + 1
                
            else:
                continue
        
        #print(i)
        f.close()
        
        
        # load the per and org classification result
        self.per_org_label()
        
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

        
 
    
    
    
    """    
    """
    def recommendationV3(self, name, n):
        self.nremd = n
        self.debugmsg('Will generate recommendations in 3 groups and ' + str(n) + ' for each group', 1)
        self.debugmsg('find the idx', 0)
        if isinstance(name, str):
             #idx = self.authors.index(name)
             name = ud.normalize('NFC',name)
             authorIdx = self.authordict.get(name)
        else:
             #idx = self.authors.index(name.decode('utf-8')) 
             name = name.decode('utf-8')
             name = ud.normalize('NFC',name)
             authorIdx = self.authordict.get(name)
             
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
        

        # randomlize the order of the recommendations
        random.shuffle(recommendations)
        
        self.result=OrderedDict([("name",name),("recommendations",recommendations)])        
        self.debugmsg('end recommendationV3 \n', 0)
        return self.result    
      
    """    
    """
    def recommendationV4(self, name, n):
        self.nremd = n
        self.debugmsg('Will generate recommendations in 3 groups and ' + str(n) + ' for each group', 1)
        self.debugmsg('find the idx', 0)
        if isinstance(name, str):
             #idx = self.authors.index(name)
             name = ud.normalize('NFC',name)
             authorIdx = self.authordict.get(name)
        else:
             #idx = self.authors.index(name.decode('utf-8')) 
             name = name.decode('utf-8')
             name = ud.normalize('NFC',name)
             authorIdx = self.authordict.get(name)
             
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
                    remdinfo = self.getremdinfoV2(i)
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
        

        # randomlize the order of the recommendations
        random.shuffle(recommendations)
        
        self.result=OrderedDict([("name",name),("recommendations",recommendations)])        
        self.debugmsg('end recommendationV4 \n', 0)
        return self.result   
  
    """
        find n nearset neighbors of point p in given space using linear search
        if n == 0, sort all the points in space
    """
    def nNNlinesearch(self, space, p, n):
        closeauthordis = []
            

        closeauthordis = pairwise_distances(space, p, metric='cosine')
        closeauthordis = closeauthordis.flatten()
            
        closeauthors = closeauthordis.argsort()
        closeauthordis.sort()
        
        if n > 0 :
            closeauthors = closeauthors[0:n]
            closeauthordis = closeauthordis[0:n]
      
        # delete myself, cuz the distance is always 0
        idx = np.where(closeauthors == self.myidx)[0][0]
        
        closeauthors = np.delete(closeauthors, idx)
        closeauthordis = np.delete(closeauthordis, idx)
        
        return (closeauthors, closeauthordis)
    


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
        trd[1] = len(tdis[tdis<t1]) + int((len(tdis[tdis<t2]) - len(tdis[tdis<t1]))/2)-1
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
        extract the detail inforamtion of the recommendation by its indx in
        the closeauthors
        ignor those unqualified ones which has few papers or not active 
        recently, and also remove known people in the mention network
    """
    def getremdinfoV2(self, clsidx):
        # get the author index from closeauthors
        remdidx = self.closeauthors[clsidx]
        
        username = self.authors[self.myidx]
        
        recentpub = self.resentpublicationsidx(remdidx)
        
        if recentpub:
            name = self.authors[remdidx]
            #[coauthors, idx, c] = self.mycoauthorsV4byidx(remdidx)
            mentionlist = self.mentionnetwork[username]
            
            if name in mentionlist:
                # skip the coauthor
                return []
            
            #
            remdid = self.id_name[name]
            
            if self.labels[remdid] == 'org':
                return []
            
            # get the recommendation's mention list
            coauthors = self.mycoauthorsV4bymentionlist(name)
            
            researchtopic = self.keywordbyidx(remdidx)
            return OrderedDict([("name",name), ("relevancy",self.closeauthordis[clsidx]),("coAuthors", coauthors),("researchTopics",researchtopic), ("recentPublications",recentpub)])
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
    def save_json(self,filename):  
        PROJECT_DIRECTORY = 'output/project/' + project_name + '/'
        with io.open(PROJECT_DIRECTORY + filename +'.json','w',encoding="utf-8") as outfile:
            outfile.write((json.dumps((self.result), ensure_ascii=False)))
   
        

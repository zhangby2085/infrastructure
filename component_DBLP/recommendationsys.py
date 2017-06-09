# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 16:38:22 2017

@author: secoder
"""
import io
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import wordnet as wn

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from wordcloudit import wordcloudit
import matplotlib.pyplot as plt
from collections import OrderedDict

import numpy as np

import json
import itertools
import codecs


class recommendationsys:
    
    def __init__(self, f_titles, f_authors, f_years, f_booktitle, n, npaper, nyear):
        self.f_titles = f_titles
        self.f_authors = f_authors
        self.f_years = f_years
        self.f_booktitle = f_booktitle
        self.clusternum = n
        
        self.npaper = npaper
        self.nyear = nyear
        self.filteredauthors=[]
        self.filteridx=[]       

        self.recommendauthor = []        
        
        self.docluster()
        self.initNLTKConditionalFreqDist()
        
        # define how much to pull close for 1st degree coauthorship connections 
        self.degree1p = 0.1
        # define how much to pull close for 2nd degree coauthorship connections 
        # based on degree1p, which is degree1p * degree2p for 2nd degree connections
        self.degree2p = 0.5
        
        # by default we will filter out those don't have publications in recent 10 years
        self.activityyear = 10

    def resentpublications(self,name):
        resentpub = []
        #titles = []
        #years = []

        if isinstance(name, unicode):
             idx = self.authors.index(name)
        else:
             idx = self.authors.index(name.decode('utf-8'))        
        
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
            resentpub.append(OrderedDict([("title",self.rawtitles[i][:-4]),("authors",authorsjson), ("year",self.years[i]),("publicationVenue",self.booktitle[i])]))

        

        return resentpub
        
    def initNLTKConditionalFreqDist(self):
        pairs=[]

        for title in self.titles:
            pairs = pairs + list(nltk.bigrams(title.split()))
    
        self.cfd = nltk.ConditionalFreqDist(pairs)
    
    def keyword(self,name):
        if isinstance(name, unicode):
             idx = self.authors.index(name)
        else:
             idx = self.authors.index(name.decode('utf-8'))
             
        content = self.authorcontents[idx].lower()
        
        # get the unique words from the content
        content = set(content.split())
        
        i = []
        for c in content:
            count = self.vectorizer.vocabulary_.get(c, 0)   
            i.append(count)
            
        i = np.array(i)
        i = i.argsort()
        content = np.array(list(content))
        content = content[i]
        content = content[-3:]
        keywords = list(reversed(content))   
        
        contentjson = []
        for topic in keywords:
            contentjson.append(OrderedDict([("topic", topic)]))
            
        # bigram keywords -------------
        content = remd.authorcontents[idx].lower().split()
        
        userpairs = list(nltk.bigrams(content))
                
        # do the same on raw titles 
        pairs=[]
        
        for title in remd.rawtitles:
            pairs = pairs + list(nltk.bigrams(title.split()))
            
        
        cfdraw = nltk.ConditionalFreqDist(pairs)
        
        keywordsraw=[]
        for p in userpairs:
            pairsdic=cfdraw[p[0]]
            n=pairsdic[p[1]]
            if n>=2:
                keywordsraw.append((p,n))
            
        uniqkeywords=set(keywordsraw)
        keywords=sorted(uniqkeywords, key=lambda keywords: keywords[1])
        
        finalkeywords=[]
        for p in keywords:
            #c=wn.synsets(p[0][1])[0].pos()
            if (p[1]>=3):
                finalkeywords.append((' '.join(p[0]),p[1],keywordsraw.count(p)))
                
        finalkeywords.reverse()
        
        for topic in finalkeywords:
            contentjson.append(OrderedDict([("topic", topic[0])]))
        
        return contentjson
        
    def mycoauthors(self, name):
        if isinstance(name, unicode):
             idx = self.authors.index(name)
        else:
             idx = self.authors.index(name.decode('utf-8'))
             
        coauthorship = self.coauthornet[idx,:]
        n = len(np.nonzero(coauthorship>0)[0])
        
        i = coauthorship.argsort()
    
        result = []
        coauthoridx = []
        coauthorcount = []        
        
        for m in range(1,n+1): 
            coauthoridx.append(i[-m])
            author = self.authors[i[-m]]
            count = coauthorship[i[-m]]
            coauthorcount.append(count)
            #result.append({'name':author, 'cooperationCount':count})
            result.append(OrderedDict([("name",author),("cooperationCount",count)]))
        return (result,coauthoridx,coauthorcount)
        
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
        a = np.array(a);
        b = np.array(b);
        return np.sqrt(sum(np.square(a - b)))
        
    """
    """
    def updatecoauthornetwork(self,net,authors,namelist):
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
        
        
    def docluster(self):
        tokenizer = RegexpTokenizer(r'\w+')

        #titledict={}
        self.rawtitles = []
        self.titles = []
        #newtitles = []
        sw = set(nltk.corpus.stopwords.words('english'))
        
        
        # filename = './CHI/CHI_titles.txt'
        
        f = codecs.open(self.f_titles,'r','utf-8')
        for line in f:   
            self.rawtitles.append(line)
            line = line.lower()
            newline=tokenizer.tokenize(line)
            # collect all the words except digtals and stopwords
            newline= ' '.join([w for w in newline if (w.lower() not in sw) & ~(self.digstring(w))])
            self.titles.append(newline)
        # end use codecs
                
        # filename = './CHI/CHI_authors.txt'
        
        self.authors = []
        self.authorcontents = []
        self.authortitlesidx = []
        self.coathors = []
        
        num = len(self.titles)
        
        # the co-author relationship matrix (2d array)
        self.coauthornet = np.zeros([num*3,num*3],dtype=int)
        
        # read years
        self.years = []
        f = codecs.open(self.f_years,'r','utf-8')
        for line in f:
            # remove the '\r\n'
            line = line[:-2]
            self.years.append(line)
            
        # read conference 
        self.booktitle = []
        f = codecs.open(self.f_booktitle,'r','utf-8')
        for line in f:
            # remove the '\r\n'
            line = line[:-2]
            self.booktitle.append(line)
        
        # use codecs
        i = 0
        f = codecs.open(self.f_authors,'r','utf-8')
        for line in f:
            # split the authors by ','
            newline = line.split(",")
            # remove the last '\n' 
            newline.remove('\r\n')
            namelist = newline
            self.coathors.append(namelist)           
            
            self.coauthornet = self.updatecoauthornetwork(self.coauthornet,self.authors,namelist)    
            
            for name in newline:
                if name not in self.authors:
                    self.authors.append(name)
                    self.authorcontents.append(self.titles[i])
                    self.authortitlesidx.append([i])
                else:    
                    idx = self.authors.index(name)
                    self.authortitlesidx[idx].append(i)
                    self.authorcontents[idx] = ' '.join([self.authorcontents[idx],self.titles[i]])
                                
            i = i + 1
            print i
        # end use codecs
        self.coauthornet = self.coauthornet[0:len(self.authors), 0:len(self.authors)];
            
                
        self.vectorizer = CountVectorizer(max_df=0.95, min_df=1,stop_words='english')
        
        X = self.vectorizer.fit_transform(self.authorcontents)
        
        Xarray = X.toarray()
        
        #plt.plot(hist)
        
        transformer = TfidfTransformer()
        
        self.tfidf = transformer.fit_transform(Xarray)
        self.tfidfarray = self.tfidf.toarray()
        
        self.featurenames = self.vectorizer.get_feature_names()
        
        # do the clustering
        
        # number of clusters
        # n = 10 
        
        self.km=KMeans(n_clusters=self.clusternum, init='k-means++',n_init=50, verbose=1)
        
        self.km.fit(self.tfidf)
        
        
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
    
        # compute the distance between the given author and all the cluster centers
        dis = []
        
        for i in self.km.cluster_centers_:
            dis.append(self.distance(featuretfidf,i))
            
        
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
        print self.authors[authorIdx] 
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
        
        while l < n*5:
            #r = max(maxdis-minclusterdis, minclusterdis-mindis)
            #r = (maxdis-mindis)/2
            r = (max(dis)-min(dis))/10*p
            
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
            
            self.closeauthors = closeidx1[closeauthorsidx1]
            
            # compute the distance between the user and all the closeauthors
            
            self.closeauthordis = []
            
            for i in self.closeauthors:
                self.closeauthordis.append(self.distance(self.tfidfarray[authorIdx],self.tfidfarray[i]))
            
            self.closeauthordis = np.array(self.closeauthordis)
            
            self.closeauthors = self.closeauthors[self.closeauthordis.argsort()]
            l = len(self.closeauthors)
            p = p+1
        
        
        self.closeauthordis.sort()
        print 'After {} got {},  recommended authors who has similar research interests: '.format(p,l)
        
        
        for i in self.closeauthors:
            self.recommendauthor.append(self.authors[i])

        # remove userself 
        self.closeauthordis = np.delete(self.closeauthordis,self.recommendauthor.index(name))
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
                
        newrecommendations = self.filteredrecommendations(n)
    
        result=OrderedDict([("name",name),("recommendations",newrecommendations)])        
        
        return result

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
        
        i = 0
        for name in self.recommendauthor:
            #coauthors = []
            #researchtopic = []
            #recentpub = []
            #coauthorsjson = []
            [coauthors, a, c] = self.mycoauthors(name)

            #for coauthor in coauthors:    
            #    coauthorsjson.append(OrderedDict([("name",coauthor)]))
        
            
            recentpub = self.resentpublications(name)

            # check if the recentpub is empty which is not active anymore
            if not recentpub:
                i = i+1
                continue
            # --    

            self.filteredauthors.append(name)            
            
            researchtopic = self.keyword(name)    
    
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
    def filteredcloseauthordis(self):
        return self.closeauthordis[self.filteridx]
    

        
"""
  Start 
"""


remdCHI = recommendationsys('./CHI/CHI_titles.txt','./CHI/CHI_authors.txt',
                            './CHI/CHI_years.txt', './CHI/CHI_booktitle.txt', 25, 3, 2013)
remd = recommendationsys('./HICSS_titles.txt','./HICSS_authors.txt',
                             './HICSS_years.txt','./HICSS_booktitle.txt',25, 3, 2013)


# make 20 recommendations to user
#user = 'Ekaterina Olshannikova'
user = 'Hannu Kärkkäinen'
recommendauthors = remd.recommendation(user,20)

for name in remd.filteredauthors:
    print name
" ---- "

#with io.open("testJson.json",'w',encoding="utf-8") as outfile:
#    outfile.write(unicode(json.dumps((recommendauthors), ensure_ascii=False)))
    
with io.open("testJson.json",'w',encoding="utf-8") as outfile:
    outfile.write((json.dumps((recommendauthors), ensure_ascii=False)))
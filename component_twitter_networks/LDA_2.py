#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 01:38:58 2018

@author: yaop
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 18:49:08 2018

@author: yaop
"""
import codecs
import spacy
import pandas as pd
import itertools as it
import codecs
import re
import urllib.request as ur
import numpy as np

from gensim.models.phrases import Phrases
from gensim.models.word2vec import LineSentence
from gensim.corpora import Dictionary, MmCorpus
from gensim.models.ldamulticore import LdaMulticore
from gensim.models import CoherenceModel

from sklearn.metrics import pairwise_distances
from config import project_name



class recommendationsys_LDA:
    def __init__(self, ngram):
        # load the spacy english model
        self.nlp = spacy.load('en')
        
        self.extrawords = ["'s", "st", "th", "’s", "-PRON-", "’", "htt", "ht", "km", "pm", "am"]
        
        # parse the latest emoji code
        html = str(ur.urlopen('http://www.unicode.org/Public/emoji/5.0/emoji-data.txt').read())
        codes=list(map(lambda x: '-'.join(['\\U'+a.zfill(8) for a in x.split('..')]).encode().decode('unicode-escape'),re.findall(r'(?<=\\n)[\w.]+',html)))
        self.emojiPattern = re.compile('['+','.join(codes)+']',flags=re.UNICODE)
        
        PROJECT_DIRECTORY = 'output/project/' + project_name

        self.f_titles = PROJECT_DIRECTORY + '/titlesLF_target.txt'

        self.f_authors = PROJECT_DIRECTORY + '/authors_target.txt'
        
        self.authorcontent_clean = {}
        
        self.ngram_bow_corpus = []
        
        self.ldavec = {}
        
        self.ngram_dictionary = None
        
        self.ngram = ngram
        self.num_topics = None

    def clean_text(self, text):
    
        # remove the 'RT' and replace '\n' to '.' 
        text = text.lower()
        #text = text.replace('RT',' ')
        text = text.replace('\n',' . ')    
    
        # this is for USC-2
        # remove emojis
        myre = re.compile(u'('
                         '@\S*\s?|#|'   # remove @ mention names and hastag sign
                         'http[s]?[:…]+\S*\s|' # remove url
                         '[-~\^\$\*\+\{\}\[\]\\\|\(\)/“"]|'
                         'rt[:]? |'
                         '…'
                         ')+', re.UNICODE)

        text = myre.sub(' ', text)
        text = self.emojiPattern.sub(' ', text)

        text = text.replace('&amp;','and')
        
        
        
                
    

        #text = ' '.join(text)
        


        
        return text


#---------------------------
# make the recommendations
#---------------------------
    def recomendation(self, username, topicn=0, list=[]):
        
        similaritis = self.ldacosinesimilarity(username,topicn)
        result=[]
        # list is empty, run on the whole dataset
        if not list:
            for key, value in sorted(similaritis.items(), key=lambda x:x[1]):
                result.append((key,value))
        else:
            for i in list:
                result.append((i,similaritis[i]))
            
            # sort the result by similarities
            result = sorted(result, key=lambda x:x[1])

#---------------------------
# load and clean the data
#---------------------------
    def loadandclean(self, n=-1):

        #authorcontent = {}

        # ------
        with codecs.open(self.f_titles, encoding='utf_8') as f_t:
            with codecs.open(self.f_authors, encoding='utf_8') as f_a:
                for l_a, l_t in zip(f_a, f_t):
                    # remove the '\n' at the end
                    key = l_a[:-1].lower()
            
                    l_t = self.clean_text(l_t)
                    if key in self.authorcontent_clean:
                        
                        self.authorcontent_clean[key].append(l_t)
                        #self.authorcontent_clean[key] = self.clean_text(value)
                    else:
                        
                        self.authorcontent_clean[key] = [l_t]
                        #self.authorcontent_clean[key] = self.clean_text(value)
                    
                    if n != -1 and len(self.authorcontent_clean) == n:
                        break
        # ---------------                
        

        for key, value in self.authorcontent_clean.items():
           self.authorcontent_clean[key] = self.lemmatized_sentence_corpus(self.authorcontent_clean[key])


    
#------------------------------------------------------
# build the trigram content based on the clean content
#------------------------------------------------------
    
    def punct_space(self, token):
        """
        helper function to eliminate tokens
        that are pure punctuation or whitespace
        """
        #return token.pos_ == 'NOUN' or token.is_punct or token.is_space or token.lemma_ in spacy.lang.en.STOP_WORDS or token.lemma_ in self.extrawords or len(str(token)) < 2
        return token.is_punct or token.is_space or token.lemma_ in spacy.lang.en.STOP_WORDS or token.lemma_ in self.extrawords or len(str(token)) < 2

    def lemmatized_sentence_corpus(self, contents):
        """
        generator function to use spaCy to parse reviews,
        lemmatize the text, and yield sentences
        """
        sentents = []
    
        for content in self.nlp.pipe(contents,batch_size=500, n_threads=8):
        
            for sent in content.sents:
                #sentents.append(u' '.join([token.lemma_ for token in sent
                #                 if not punct_space(token)]))
                #sentents.append([token.lemma_ for token in sent
                #                 if not punct_space(token)])
                tokens = []
                for token in sent:
                    if self.punct_space(token):
                        continue
                
                    #if token.lemma_ == '-PRON-':
                    #    token.lemma_ = token.lower_
                    tokens.append(token.lemma_)
                
                sentents.append(tokens)
                    
        return sentents

    """
    prepare the parameters for lda
    """
    def ldainit(self):
        
#        self.num_topics = num_topics
#        ngram = self.ngram
#        # if ngram_bow_corpus is empty, build it first
#        if not self.ngram_bow_corpus: 
        
        self.user_sentences = self.authorcontent_clean
        self.user_bigramsentences = {}
        self.all_sentences = []
        self.all_bigram_sentences = []
        
        sentences = list(self.authorcontent_clean.values())
        self.all_sentences = [item for sublist in sentences for item in sublist]
        
        # buld bigram model 
        if self.ngram == 2:
            self.bigram_model = Phrases(self.all_sentences)
            for user,content in self.user_sentences.items():
                bigram_s = []
                for s in content:
                    bigram_s.append(self.bigram_model[s])
                self.user_bigramsentences[user] = bigram_s
                self.all_bigram_sentences += self.user_bigramsentences[user]
                
            
            
    def trainlda(self, topics_n = 10):
        self.num_topics = topics_n
        
        alltexts = []
        for name,sentences in self.user_sentences.items():
            sentences = [item for sublist in sentences for item in sublist]
            alltexts.append(sentences)
        
        
#        if self.ngram_dictionary == None:
#            if self.ngram == 1:
#                self.ngram_dictionary = Dictionary(self.all_sentences)
#            elif self.ngram == 2:
#                self.ngram_dictionary = Dictionary(self.all_bigram_sentences)
#                
        if self.ngram_dictionary == None:
            if self.ngram == 1:
                self.ngram_dictionary = Dictionary(alltexts)
            elif self.ngram == 2:
                self.ngram_dictionary = Dictionary(alltexts)
                
            # filter tokens that are very rare or too common from
            # the dictionary (filter_extremes) and reassign integer ids (compactify)
            self.ngram_dictionary.filter_extremes(no_below=10, no_above=0.8)
            self.ngram_dictionary.compactify()


#        if self.ngram == 1:
#            sentences = self.all_sentences
#        elif self.ngram == 2:
#            sentences = self.all_bigram_sentences
            
#        ngram_bow_corpus = []
#        for sentence in sentences:
#            ngram_bow_corpus.append(self.ngram_dictionary.doc2bow(sentence))
#
#
#        self.lda = LdaMulticore(ngram_bow_corpus,
#                           num_topics = topics_n,
#                           id2word=self.ngram_dictionary,
#                           workers=3)
        

            
        ngram_bow_corpus = []
        for sentence in alltexts:
            ngram_bow_corpus.append(self.ngram_dictionary.doc2bow(sentence))


        self.lda = LdaMulticore(ngram_bow_corpus,
                           num_topics = topics_n,
                           id2word=self.ngram_dictionary,
                           workers=3)    
        
        
                # calculate the cohe
        topics=[]

        for i in range(self.lda.num_topics):
            terms = []
            for n in self.lda.show_topic(i):
                terms.append(n[0])
            topics.append(terms)
        
        cm_umass = CoherenceModel(topics=topics, corpus=ngram_bow_corpus, dictionary=self.ngram_dictionary, coherence='u_mass')
        cm_cv = CoherenceModel(topics=topics, texts=alltexts, dictionary=self.ngram_dictionary, coherence='c_v')
        cm_cuci = CoherenceModel(topics=topics, texts=alltexts, dictionary=self.ngram_dictionary, coherence='c_uci')
        cm_cnpmi = CoherenceModel(topics=topics, texts=alltexts, dictionary=self.ngram_dictionary, coherence='c_npmi')

        return topics_n, cm_umass.get_coherence(), cm_cv.get_coherence(),cm_cuci.get_coherence(),cm_cnpmi.get_coherence()

        
    def explore_topic(self, topic_number, topn=25):
        """
        accept a user-supplied topic number and
        print out a formatted list of the top terms
        """
        
        print(u'{:20} {}'.format(u'term', u'frequency') + u'\n')

        for term, frequency in self.lda.show_topic(topic_number, topn):
            print(u'{:20} {:.3f}'.format(term, round(frequency, 3)))



    def runlda(self, username):
        
        if self.ngram == 1:
            user_sentences = self.user_sentences[username]
        elif self.ngram == 2:
            user_sentences = self.user_bigramsentences[username]
        
        # flat the list of list into single list
        user_sentences = [item for sublist in user_sentences for item in sublist]
        user_bow = self.ngram_dictionary.doc2bow(user_sentences)

        user_lda = self.lda[user_bow]

        #user_lda = sorted(user_lda, key=lambda x:-x[1])
        
        return user_lda

    """
    compute the lda topic vec for every one 
    """
    def runldavec(self):
        if not self.ldavec:
            for key, value in self.user_sentences.items():
                vec = np.zeros(self.num_topics)
                result = self.runlda(key)
                for i in result:
                    vec[i[0]] = i[1]
                self.ldavec[key] = vec
                
            
    """
    """
    def runtopntopic(self, n):
        self.topntopics = []
        
        for key, value in self.ldavec.items():
            idx = value.argsort()
                
            self.topntopics += list(idx[-n:])
        
        self.topntopics = list(set(self.topntopics))
    
    """
    compute the lda cosine similarity between a given user and the rest users
    """
    def ldacosinesimilarity(self, username, topn=0):
        if username not in self.authorcontent_clean:
            print('The user cannot find')
            return
        if topn < 0:
            print('topn should be >= 0')
            return
        
        topn = int(topn)
        
        cosinesimilaritydic = {}
        
        if not self.ldavec:
            self.runldavec()
        
        if topn == 0:
            usertopicvec = self.ldavec[username]
        else:
            self.runtopntopic(topn)
            usertopicvec = self.ldavec[username][self.topntopics]
            
        for key, value in self.ldavec.items():
            if key != username:
                if topn == 0:
                    pairtopicvec = value
                else:
                    pairtopicvec = value[self.topntopics]
                cosinesimilarity = pairwise_distances(np.array(usertopicvec).reshape(1,-1),np.array(pairtopicvec).reshape(1,-1), metric='cosine')[0][0]
                cosinesimilaritydic[key] = cosinesimilarity
                
        return cosinesimilaritydic
#--------end ----


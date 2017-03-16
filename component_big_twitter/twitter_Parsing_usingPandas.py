# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 10:35:57 2017

@author: guptaj
"""
import pandas as pd
import sframe
# tweet-preprocessor 0.5.0
import preprocessor as p
import networkx as nx
#sf_full = sframe.SFrame.read_csv('data/01-raw/mini_sample.csv', header=False)
pd_full = pd.read_csv('data/01-raw/mini_sample.csv',encoding = 'utf-8')
#
#sf_full = sframe.SFrame(data = pd_full)
#p = ttp.Parser()
#print sf_full.head()
#print sf_full['X5'][1]
sample_tweet1 = "@bärnettedmond, you now support #IvoWertzel's tweet parser! https://github.com/edburnett/"
sample_tweet2 = "@bärnettedmond, @jysh support #IvoWertzel's tweet parser! https://github.com/edburnett/"
sample_tweet3 = u'RT @aukia: (In Finnish) Ilmoittaudu 28.5 ""Kuinka ostaa ketterästi julkishallinnossa"" aamiaisseminaariin. @codento,Hansel,MML,HY http://t.co…"
#tweet_encoded = sample_tweet.encode('utf-8')
#tweet = sf_full['X5'][11]
#result = p.parse(tweet_encoded)
#print p.clean('Preprocessor is #awesome 👍 https://github.com/s/preprocessor')
#print p.clean(tweet)
#print result.userss
#parsed_tweet = p.parse(tweet)
# 
#print parsed_tweet.mentions
#print parsed_tweet.hashtags
#mention = p.parse(sample_tweet2).mentions
#print mention[0].match.strip('@') 
#for indvmention in mention:
#    print indvmention.match
             

             
def get_usermentions(singletweet):
    return p.parse(singletweet).mentions

#print get_usermentions(sample_tweet2)  
#print sf_full.dtype()         
#sf_full_2 = sf_full['X5'].apply(get_usermentions)

#sf_full_trial = sf_full[[14]]

#print len(sf_full)
#print range(len(sf_full))

implicit_network = nx.DiGraph()

for i in range(len(pd_full)):
    try:
        mentioned = get_usermentions(pd_full['X5'][i])
#        mentioned = p.parse(sf_full['X5'][i]).mentions
        if mentioned != None:
            for indvmention in mentioned:
                print pd_full['X4'][i],indvmention.match.strip('@')
                if not implicit_network.has_edge(pd_full['X4'][i],indvmention.match.strip('@')):
                    implicit_network.add_edge(pd_full['X4'][i],indvmention.match.strip('@'), weight = 0)
                implicit_network[pd_full['X4'][i]][indvmention.match.strip('@')]['weight'] += 1
                    
    except Exception as e:
        print i
        print "the error is"
        print e

nx.readwrite.gexf.write_gexf(implicit_network,'data/02-network/trial_network2.gexf', encoding='utf-8')
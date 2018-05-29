#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 21:13:05 2018

@author: yaop
"""
from LDA_2 import recommendationsys_LDA

from config import project_name

from mention_network import mention_network

#-------
import io
import json

import numpy as np

def save_json(var, filename):
    with io.open(filename+'.json','w',encoding="utf-8") as outfile:
        outfile.write((json.dumps(var, ensure_ascii=False)))


def sort_by_distance(all_distance, user_name, user_list):
    sort_dis = {}
    for u in user_list:
        sort_dis[u] = all_distance[user_name][u]
    
    sort_dis = sorted(sort_dis.items(), key=lambda x:x[1])
    return sort_dis

PROJECT_DIRECTORY = 'output/project/' + project_name

# use the unigram, 2 to use bigram
recsys_lda_unigram = recommendationsys_LDA(1)

# load all the data
recsys_lda_unigram.loadandclean()

# lda init
recsys_lda_unigram.ldainit()

# train 70 topics
recsys_lda_unigram.trainlda(70)

recsys_lda_unigram.runldavec()
  
    
# ------------------
# compute all the lda cosines similarity(distance) between 
# each user and the others, and compute the min, max and mean distance for each
# ------------------
all_distance = {}
stat_distance = [[],[],[]]
for key in recsys_lda_unigram.ldavec:
    all_distance[key] = recsys_lda_unigram.ldacosinesimilarity(key,0)
    dis = list(all_distance[key].items())
    dis = [x[1] for x in dis]
    stat_distance[0].append(np.min(dis))
    stat_distance[1].append(np.max(dis))
    stat_distance[2].append(np.mean(dis))
    
# ------------------
# get the 3 groups candidate for each user from mention network perspective
# __________________
mn = mention_network("./output/project/Tampere3_all_1/mention_network_ristrict_per_org.gexf",partition_resolution=0.85)
results = mn.get_recommendations()


user_statistic = {}
for key in all_distance:
    if key not in mn.all_paths:
        continue
    results = mn.get_recommendations(key)
    g1 = sort_by_distance(all_distance,key,results['close'])
    g2 = sort_by_distance(all_distance,key,results['mid'])
    g3 = sort_by_distance(all_distance,key,results['far'])
    
    if len(g1) >=3 and len(g2) >=3 and len(g3) >=3:
        valid = 1
        dis = [x[1] for x in g1]
        mean_all_1 = np.mean(dis)
        mean_top_1 = np.mean(dis[0:3])
        
        dis = [x[1] for x in g2]
        mean_all_2 = np.mean(dis)
        mean_top_2 = np.mean(dis[0:3])
        
        dis = [x[1] for x in g3]
        mean_all_3 = np.mean(dis)
        mean_top_3 = np.mean(dis[0:3])
        
        mean1 = np.mean([x[1] for x in g1[0:3]])
        mean2 = np.mean([x[1] for x in g2[0:3]])
        mean3 = np.mean([x[1] for x in g3[0:3]])
    else:
        valid = 0
        mean1 = 0
        mean2 = 0
        mean3 = 0
    

    
    std = np.std([mean1,mean2,mean3])
    user_statistic[key] = [valid, mean1, mean2, mean3, std, g1[0:3], g2[0:3], g3[0:3],
                   mean_all_1, mean_all_2, mean_all_3, mean_top_1,mean_top_2,mean_top_3]


save_json(user_statistic, 'user_statistic')

# ------------------
# Following are some temporary code for plot and test run  
# ------------------

'''

from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
# valid users
valid_users = {}

for k,v in user_statistic.items():
    if v[0] == 1:
        valid_users[k] = v

# find threshold by mean distance
thresh = threshold_otsu(np.sort(stat_distance[2]))

# mean dis for each user
user_mean = {}
for k,v in all_distance.items():
    dis = list(v.items())
    dis = [x[1] for x in dis]
    user_mean[k] = np.mean(dis)
    
# filter the valid users by thresh
filter_user = {}

for k,v in user_mean.items():
    if v <= thresh:
        if k in user_statistic:
            filter_user[k] = user_statistic[k]

# mean of mean for each group
user_groupmean = [[],[],[]]
for k, v in valid_users.items():
    user_groupmean[0].append(v[8])
    user_groupmean[1].append(v[9])
    user_groupmean[2].append(v[10])
    
mg1 = [ float("{0:.2f}".format(x)) for x in user_groupmean[0] ]
mg2 = [ float("{0:.2f}".format(x)) for x in user_groupmean[1] ]
mg3 = [ float("{0:.2f}".format(x)) for x in user_groupmean[2] ]

plt.hist(mg1, np.unique(mg1))

# ------------------
    
# ------------------
import matplotlib.pyplot as plt
d1 = [ float("{0:.4f}".format(x)) for x in stat_distance[2] ]

beingsaved = plt.figure()
plt.hist(d1, np.unique(d1))
beingsaved.savefig('mean_distance_all_hist.eps', format='eps', dpi=1000)

beingsaved = plt.figure()
plt.plot(np.sort(stat_distance[2]))
beingsaved.savefig('mean_distance_all_plot.eps', format='eps', dpi=1000)

beingsaved = plt.figure()
plt.plot(np.sort(list(all_distance['jnkka'].values())))
beingsaved.savefig('jnkka_distance.eps', format='eps', dpi=1000)

# ------------------
    
# ---------------------------
a=recsys_lda_unigram.ldacosinesimilarity('jnkka',10)

b=[]

for key, value in sorted(a.items(), key=lambda x:x[1]):
    b.append((key,value))
#----------------------------
import matplotlib.pyplot as plt
n = 20
c=[]
for key, value in recsys_lda_unigram.ldavec.items():
    idx = value.argsort()

    c += list(idx[-n:])

plt.hist(c,np.unique(c))
    
#---------------------------
import matplotlib.pyplot as plt
c=[x[1] for x in b]
plt.plot(c)

topicdistribution = recsys_lda_unigram.runlda('jnkka')

topicdistribution = sorted(topicdistribution, key=lambda x:-x[1])

for i in topicdistribution:
    
    print(i)
    print('|____')
    recsys_lda_unigram.explore_topic(i[0],15)
    print('\n')
    
import numpy as np
coherence = []
for i in np.arange(5,505,5):
    coherence.append(recsys_lda_unigram.trainlda(i))
    
    
    


topics_n = []
umass = []
cv = []
cuci = [] 
cnpmi = []
for m in coherence:
    topics_n.append(m[0])
    umass.append(m[1])
    cv.append(m[2])
    cuci.append(m[3])
    cnpmi.append(m[4])
    

import matplotlib.pyplot as plt
import numpy as np

plt.plot(topics_n, umass)
plt.plot(topics_n, cv)
plt.plot(topics_n, cuci)
plt.plot(topics_n, cnpmi)
plt.show()

# save the figure in high resolution
beingsaved = plt.figure()
plt.plot(topics_n, cnpmi)

beingsaved.savefig('cnpmi_only_noun.eps', format='eps', dpi=1000)

'''



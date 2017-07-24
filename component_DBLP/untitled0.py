# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 14:17:11 2017

@author: secoder
"""


import io
import json

from recommendationsys import recommendationsys


# initiate the recommendation system for CHI
# 10 is the cluster number(useless in current version, just ignor)
# 3 is the threshold for the number of the publications of a recommendation
# 2013 is the threshold for the most recent year of the publication of a recommendation
remdCHI = recommendationsys('./CHI/CHI_titles.txt','./CHI/CHI_authors.txt',
                            './CHI/CHI_years.txt', './CHI/CHI_booktitle.txt', 10, 3, 2013)

remd = recommendationsys('./HICSS_titles.txt','./HICSS_authors.txt',
                             './HICSS_years.txt','./HICSS_booktitle.txt',10, 3, 2013)

remdKATE = recommendationsys('./KATE/WWIC_ICC_GLOBECOM_titles.txt','./KATE/WWIC_ICC_GLOBECOM_authors.txt',
                             './KATE/WWIC_ICC_GLOBECOM_years.txt','./KATE/WWIC_ICC_GLOBECOM_booktitle.txt',10, 3, 2013)


# users
user1 = 'aaa'
user2 = 'bbb'
user3 = 'ccc'

user4 = 'ddd 001'
user5 = 'eee'

# by default the system will generate 3 groups recommendations
# example: remd.recommendationV3(user2,3) will return 9 recommendations 
# for user2 and 3 in each group 

recommendauthors2 = remd.recommendationV3(user2,3)
recommendauthors3 = remdCHI.recommendationV3(user3,3)
recommendauthors4 = remdKATE.recommendationV3(user4,3)

# print the recommendations and their relevcancy (distance) for user2
for author in recommendauthors2['recommendations']:
    print author['name'] + ' ' + str(author['relevancy'])

# save the recommendations to json file aaa.json
remd.save_json('aaa')

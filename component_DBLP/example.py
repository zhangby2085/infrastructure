# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 14:17:11 2017

@author: secoder
"""

from recommendationsys import recommendationsys

user1 = 'Mikko Valkama'
user2 = 'Aino Ahtinen'
user3 = 'Heli Väätäjä'
user4 = 'Harri Siirtola'
user5 = 'Yevgeni Koucheryavy'

remd = recommendationsys('./Yevgeni/titles.txt','./Yevgeni/authors.txt',
                            './Yevgeni/years.txt', './Yevgeni/booktitles.txt', 10, 3, 2013)

recommendauthors = remd.recommendationV3(user5,3)

remd.save_json(user5)

# print the recommendations and relevancy
for author in recommendauthors['recommendations']:
    print author['name'] + ' ' + str(author['relevancy'])
    

# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 14:17:11 2017

@author: secoder
"""
import sys
import subprocess

from recommendationsys import recommendationsys

def runremd(user, arg):
    cmd=['./extractconferencedata.sh']
    subprocess.call(cmd+arg)

    f_titles='./'+arg[0]+'/titles.txt'
    f_authors='./'+arg[0]+'/authors.txt'
    f_years='./'+arg[0]+'/years.txt'
    f_booktitles='./'+arg[0]+'/booktitles.txt'


    remd = recommendationsys(f_titles,f_authors, f_years, f_booktitles, 10, 3, 2013)

    recommendauthors = remd.recommendationV3(user,3)

    remd.save_json('./'+arg[0]+'/'+user)

    # print the recommendations and relevancy
    for author in recommendauthors['recommendations']:
        print author['name'] + ' ' + str(author['relevancy'])
    


if len(sys.argv) < 2:
    print 'Usage: {0} filename or {0} username folder conference1 conference2 conference3...'.format(sys.argv[0])
    print 'Example: {0} list,  {0} "Jari Kangas" Jari  "CHI Extended Abstracts"  ETRA CHI'.format(sys.argv[0])
    print 'The file should contain the paremters line by line and seperate by "," without space in between'
    exit()    

if len(sys.argv) == 2:
    with open('./'+sys.argv[1], 'r') as f:
        for line in f:
            args=line[:-1].split(',')
            user=args[0]
            arg=args[1:]
            runremd(user,arg)
else:
    user=sys.argv[1]
    arg=sys.argv[2:]


    print "This is the name of the script: ", sys.argv[0]
    print "Number of arguments: ", len(sys.argv)
    print "The arguments are: " , str(sys.argv)

    runremd(user,arg)

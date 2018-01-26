#!/bin/bash
PROJECT=`grep project_name config.py | cut -d "'" -f 2`
PROJECTDIR="./output/project/${PROJECT}"
# there are ^M in titles.txt, this cmd is to remove it

# doesn't work in OSX
#sed -e "s/\r/ /g" ${PROJECTDIR}/titles.txt > ${PROJECTDIR}/titlesLF.txt
sed -e "s// /g" ${PROJECTDIR}/titles.txt > ${PROJECTDIR}/titlesLF.txt

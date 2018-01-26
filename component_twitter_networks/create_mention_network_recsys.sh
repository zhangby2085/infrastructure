#!/bin/bash
PROJECT=`grep project_name config.py | cut -d "'" -f 2`
PROJECTDIR="./output/project/${PROJECT}"

#grep target ${PROJECTDIR}/mention_network.gexf | cut -d '"' -f 4,6,8 --output-delimiter=' ' > ${PROJECTDIR}/mention_network.txt

# on OSX, no support for --output-delimiter in cut 
grep target ${PROJECTDIR}/mention_network.gexf | cut -d '"' -f 4,6,8 > ${PROJECTDIR}/mention_network.txt

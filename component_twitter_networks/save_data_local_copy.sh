#!/bin/bash

PROJECT=`grep project_name config.py | cut -d "'" -f 2`
DB=`grep db_name config.py | cut -d "'" -f 2`
PROJECTDIR="./output/project/${PROJECT}"

#FOLLOWERSID_FILE="$PROJECTDIR/project_followersID.txt"
#FOLLOWERSID_FILE="$PROJECTDIR/authors_id.txt"
FOLLOWERSID_FILE="$PROJECTDIR/authors.txt"

#FOLLOWERS=`cat $FOLLOWERSID_FILE`
FOLLOWERS=`cat $FOLLOWERSID_FILE | uniq `

OUTPUTDIR="$PROJECTDIR/data_by_id"
mkdir -p $OUTPUTDIR

for i in $FOLLOWERS; do
    #continue   
    DONE=`ls $OUTPUTDIR | cut -d "." -f 1 | grep -w ${i}`
    #echo $DONE  
  
    if [ ! -z "$DONE" ]; then
     #   echo DONE
        continue
    fi

    #echo process
    #break
    F="$OUTPUTDIR/$i.json"
    #QUERY="{ \"user.id\": $i }"
    QUERY="{ \"user.screen_name\": \"$i\" }"
    nohup mongoexport -d ${DB} -c tweets -q "${QUERY}" --out $F --quiet > /dev/null 2>&1 &
    # keep 100 mongoexport 
    PROCESS=`ps aux | grep mongoexport | wc -l`

    while [ "$PROCESS" -gt 100 ]; do
        sleep 5s
        PROCESS=`ps aux | grep mongoexport | wc -l`
    done

done


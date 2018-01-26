#!/bin/bash
PROJECT=`grep project_name config.py | cut -d "'" -f 2`
PROJECTDIR="./output/project/${PROJECT}"

NUM=`grep 'threshold_num = ' config.py | cut -d"'" -f2`
LAN=`grep 'lan = ' config.py | cut -d"'" -f2`

cat ${PROJECTDIR}/authors.txt |  sort | uniq -c | awk -v a="$NUM" ' $1 > a ' | awk {'print $2'} >  ${PROJECTDIR}/target_author_list.txt
#TARGET_AUTHORS=$(cat ${PROJECTDIR}/authors.txt |  sort | uniq -c | awk -v a="$NUM" ' $1 >= a ' | awk {'print $2'})
TARGET_AUTHORS=$(cat ${PROJECTDIR}/authors.txt | uniq -c | awk -v a="$NUM" ' $1 >= a ' | awk {'print $2'})

#declare -a IDX

# create author_lan.txt
paste ${PROJECTDIR}/authors.txt ${PROJECTDIR}/lan.txt > ${PROJECTDIR}/authors_lan.txt

for x in $TARGET_AUTHORS;do
	RESULT=(`grep -n -E -w $x.$LAN ${PROJECTDIR}/authors_lan.txt | awk '{print $1}'`)

	if [ -z "$RESULT" ]; then
		C=0
	else
		C=${#RESULT[@]}
	fi

	if [ "$C" -ge "$NUM" ]; then
		awk -F: '{print $1}' <<<  "$(printf "%s\n" "${RESULT[@]}")" >> ${PROJECTDIR}/2
	fi
	#break /debug
done

#printf "%s\n" "${IDX[@]}"

awk  -v dir="$PROJECTDIR" 'FNR==NR{line[$0]=$0; next} FNR in line' ${PROJECTDIR}/2 ${PROJECTDIR}/cleantitles.txt >> ${PROJECTDIR}/cleantitles_target.txt
awk  -v dir="$PROJECTDIR" 'FNR==NR{line[$0]=$0; next} FNR in line' ${PROJECTDIR}/2 ${PROJECTDIR}/authors.txt >> ${PROJECTDIR}/authors_target.txt
awk  -v dir="$PROJECTDIR" 'FNR==NR{line[$0]=$0; next} FNR in line' ${PROJECTDIR}/2 ${PROJECTDIR}/authors_id.txt >> ${PROJECTDIR}/authors_id_target.txt
awk  -v dir="$PROJECTDIR" 'FNR==NR{line[$0]=$0; next} FNR in line' ${PROJECTDIR}/2 ${PROJECTDIR}/years.txt >> ${PROJECTDIR}/years_target.txt
awk  -v dir="$PROJECTDIR" 'FNR==NR{line[$0]=$0; next} FNR in line' ${PROJECTDIR}/2 ${PROJECTDIR}/venues.txt >> ${PROJECTDIR}/venues_target.txt

#!/bin/bash

function extractlines {
	#echo $3
	sed -n -e "$1 , $2 p" ./title_author_conf_year/title.txt >> ./"$4"/"${3}"_titles.txt
	sed -n -e "$1 , $2 p" ./title_author_conf_year/authors.txt >> ./"$4"/"${3}"_authors.txt
	sed -n -e "$1 , $2 p" ./title_author_conf_year/year.txt >> ./"$4"/"${3}"_years.txt
	sed -n -e "$1 , $2 p" ./title_author_conf_year/booktitleLF.txt >> ./"$4"/"${3}"_booktitle.txt
	#echo "new line" >> ${3}_year.txt
}

# check the input args and print the usage
if [ "$#" -lt 2 ]; then
        echo "Usage: $0  name conference1 conference2 conference3 ... "
        echo "Example: $0 Mike \"CHI\" \"HICSS\" "
	exit
fi

echo "System init ...."

# check if all the files are there
if [ ! -f ./DBLPdataset/dblp.dtd ] && [ ! -f ./DBLPdataset/dblp.xml ]; then
        printf "File dblp.xml, dblp.dtd not found, please check ./DBLPdataset directory\n"
        exit
fi

# check if the titles, authors, years, booktitiles are ready

if [ ! -d ./title_author_conf_year ]; then
        printf "Create direcotry ./title_author_conf_year\n"
        mkdir title_author_conf_year
        python ./getConf.py
        python ./getTitle.py
        python ./getAuthor.py
        python ./getYear.py
        tr -d '\r' < booktitle.txt >  booktitleLF.txt
        rm booktitle.txt
        mv *txt title_author_conf_year
fi

# ------------------------------------
# start extract target conference data
# -----------------------------------

if [ ! -f ./title_author_conf_year/booktitleLF.txt ]; then
	printf "File ./title_author_conf_year/booktitleLF.txt not found\n"
	exit
fi
                
printf "Start extract conference data:\n\n"

l=0
for x in "$@"; do
	# the first arg is the user name, make a dir for him/her
	if [ $l == 0 ]; then
		username=$x
		mkdir "$username"
		l=1
		continue
	fi

	echo "Processing $x ..."
	result=`grep -n ^"$x"$ ./title_author_conf_year/booktitleLF.txt | cut -f1 -d:` 
	#result=`grep -nw "$x" ./title_author_conf_year/booktitleLF.txt | cut -f1 -d:` 
        if [ -z "$result" ]; then
                printf "No conference found: $x\n"
                continue
        fi

	echo "Done for $x"

	if [ $l == 1 ]; then
		#printf "1st conference\n"
		idx=$result
		l=2
	else
		idx="$idx"$'\n'"$result"
	fi
done

echo "$idx" > ./"$username"/idx2
        #result=`grep -w $1 ./booktitleLF.txt | sort | uniq`
        #printf "$idx\n"
if [ -z "$idx" ]; then
	printf "No related conference found, please check the conference name and try it again\n"
	exit
fi

printf "Continue [Y/y]"

read input

if [ "$input" != "Y" ] && [ "$input" != "y" ]; then
	printf "$input Exit\n"
        exit
fi

printf "start processing....\n"

awk 'FNR==NR{line[$0]=$0; next} FNR in line' ./"$username"/idx2 ./title_author_conf_year/title.txt >> ./"$username"/titles.txt
awk 'FNR==NR{line[$0]=$0; next} FNR in line' ./"$username"/idx2 ./title_author_conf_year/authors.txt >> ./"$username"/authors.txt
awk 'FNR==NR{line[$0]=$0; next} FNR in line' ./"$username"/idx2 ./title_author_conf_year/year.txt >> ./"$username"/years.txt
awk 'FNR==NR{line[$0]=$0; next} FNR in line' ./"$username"/idx2 ./title_author_conf_year/booktitleLF.txt >> ./"$username"/booktitles.txt

# remove the " in the author.txt
sed -i -e 's/"//g' ./"$username"/authors.txt

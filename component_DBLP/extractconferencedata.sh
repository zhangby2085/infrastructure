#!/bin/bash
#idx=(2 3 4 6 7 9 12 13)

function extractlines {
	#echo $3
	sed -n -e "$1 , $2 p" ../title_author_conf_year/title.txt >> "${3}"_titles.txt
	sed -n -e "$1 , $2 p" ../title_author_conf_year/authors.txt >> "${3}"_authors.txt
	sed -n -e "$1 , $2 p" ../title_author_conf_year/year.txt >> "${3}"_years.txt
	sed -n -e "$1 , $2 p" ../title_author_conf_year/booktitle.txt >> "${3}"_booktitle.txt
	#echo "new line" >> ${3}_year.txt
}



if [[ -n "$1" ]]; then
        printf "Here are the conferences contain $1\n"
        grep -w $1 ../title_author_conf_year/booktitle.txt | sort | uniq
        printf "Continue [Y/y]"

        read input

        if [ "$input" != "Y" ] && [ "$input" != "y" ]; then
		printf "$input Exit\n"
                exit
	fi

	idx=`grep -nw $1 ../title_author_conf_year/booktitle.txt | cut -f1 -d:` 
else
    	printf "Usage: $0 Conference Name\nexample: $0 HICSS\n"
	exit
fi

printf "start processing....\n"

c=1
s=0
f=0

for i in $idx; do
	if [ $s -eq 0 ]; then
		s=$i
		continue
	fi

	if [ $(($i-$s)) -eq $c ]; then 
		c=$(($c+1))	
	else
		f=$(($c+$s-1))
		#echo $s $f 
		extractlines $s $f $1
		s=$i
		c=1
	fi
done

#echo $s $i
extractlines $s $i $1

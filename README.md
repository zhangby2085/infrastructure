Example to get all the records from HICSS

1. run getConf.py to generate the booktitle.txt which contains all the conference records
2. Check how many lines of HICSS records

    grep -ni HICSS ./booktitle.txt | wc -l

    11882
3. Check the start and end line of the HICSS records in the booktitle.txt

    grep -ni HICSS ./booktitle.txt | head -1

    3055298:HICSS

    grep -ni HICSS ./booktitle.txt | tail -1

    3067179:HICSS
4. The result(11882 = 3067179 - 3055298) indicats the HICSS records fall into the region [3055298,3067179]. Run getAuthor.py, getTitle.py, getYear.py and Extract the HICSS records from title.txt, authors.txt, year.txt

    sed -n -e '3055298,3067179 p' title.txt > HICSS_titles.txt

    sed -n -e '3055298,3067179 p' authors.txt > HICSS_authors.txt

    sed -n -e '3055298,3067179 p' year.txt > HICSS_years.txt


  NOTE:

  The line of titles is one more than the line of authors and year. The reason is that one title record contains Russian character and is splitted into two lines. Has to fix it by hand to merge those two lines

  sed -n -e '330948 p'  title.txt

# infrastructure

## Activity streams

* https://twitter.com/jnkka/status/826727865093861376

## Data sources

System-level view/Full view
* [DBLP](http://dblp.uni-trier.de/)
* [Virta REST](https://confluence.csc.fi/display/VIR/REST-lukurajapinta)
* Twitter
* [Scraping the ACM Library](https://doi.org/10.1145/511144.511146)

Profile data
* LinkedIn
* [ORCID](https://orcid.org/)

## Analytics

Network analytics
* [NetworkX](https://networkx.github.io/)
* [SNAP.py](http://snap.stanford.edu/snappy/index.html)
* [EDENetworks](http://becs.aalto.fi/edenetworks/)

Processing pipelines

* [Luigi](https://github.com/spotify/luigi) seems interesting

## Visualization

Layout
* Option to use Gephi Toolkit to process layouts (this insists on using a "big machine" rather than a cluster)

## UI/application ("data product")

* Could we use [Firebase](http://firebase.google.com) as backend?
* [mLabs](https://mlab.com/)
* [Sigma.js](http://sigmajs.org/), [D3.js](https://bl.ocks.org/mbostock/4062045), or [GEXF.js](https://github.com/raphv/gexf-js) for visualizing the network

# HPC infrastrucure

Two main sources for high performance computing resources are available: CSC and TCSC

* https://wiki.tut.fi/TCSC/Merope
* [Webinar: Creating a Virtual Machine in cPouta](https://www.youtube.com/watch?v=CIO8KRbgDoI&index=5&list=PLD5XtevzF3yHz-4PF_qBN3g26K7RzLO_c)

JH:

salloc allows you to access an interactive environment
set | grep SLURM
ssh nodeid
logout
exit

salloc -p bigmem

top

sinfo

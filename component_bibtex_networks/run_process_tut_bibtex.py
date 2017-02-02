# run_process_tut_bibtex.py

import bibtexparser
from pprint import pprint
import networkx as nx
import time

with open('data/01-raw/full-texts-at-tut.bib') as bibtex_file:
  bibtex_str = bibtex_file.read()

# bib_database = bibtexparser.loads(bibtex_str)
# print(bib_database.entries)
# for entry in bib_database.entries:
#   pprint(entry)

#   time.sleep(10)


from pybtex.database import parse_file 


def author_name(person):
  nameparts = list()
  nameparts.extend(person.first_names)
  nameparts.extend(person.last_names)
  name = ' '.join(nameparts)
  name = name.replace('{', '')  
  name = name.replace('}', '')  
  return name 

network = nx.Graph()
db = parse_file('data/01-raw/full-texts-at-tut.bib')

for entry in db.entries:
  pprint(entry)
  print len(db.entries[entry].persons['author']), 'authors'
  for author in db.entries[entry].persons['author']:
    print author
    if not network.has_node(author_name(author)):
      network.add_node(author_name(author))

  for index_a, author_a in enumerate(db.entries[entry].persons['author']):
    for index_b, author_b in enumerate(db.entries[entry].persons['author'][index_a+1:]):
      if not network.has_edge(author_name(author_a), author_name(author_b)):
        network.add_edge(author_name(author_a), author_name(author_b), weight=0)
      network[author_name(author_a)][author_name(author_b)]['weight'] += 1 

  # time.sleep(10)

nx.readwrite.gexf.write_gexf(network, 'data/03-network/co-authors.gexf', 
  version='1.2draft', encoding='utf-8')

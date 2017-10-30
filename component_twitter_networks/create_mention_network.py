import json
#import ijson
import networkx as nx
from config import id_or_screen_name,project_name,db_host,db_port,db_name,followerID_collection_name,id_name_collection_name,tweet_collection_name,tweet_num
from pymongo import MongoClient

PROJECT_DIRECTORY = 'output/project/' + project_name

NETWORK_FILE_NAME = PROJECT_DIRECTORY + '/mention_filtered_intranetwork_retweet_and_others_v2.gexf'

TWEETS_COUNT = PROJECT_DIRECTORY + '/followers_tweets_count.json'
MENTIONNAME_COUNT = PROJECT_DIRECTORY + '/followers_mentionname_count.json'


# connect to localhost mongodb
client = MongoClient(db_host, db_port)

# set the database name
db = client[db_name]

# retrive the id name mapping
result = db[id_name_collection_name].find({},{"_id":0})

ID_NAME = {}
for doc in result:
    ID_NAME[doc['id']] = doc


GRAPH = nx.DiGraph()

def add_connection(user_from, user_to):
    if not GRAPH.has_edge(user_from, user_to):
        # if connection does not yet exists, create one
        GRAPH.add_edge(user_from, user_to, weight=0)
    # add +1 to connection weight
    GRAPH[user_from][user_to]['weight'] += 1


def is_in_follower_list(user_id):
    if user_id in ID_NAME:
        return True
    else:
        return False


if is_in_follower_list(923913109080813568):
    print(11111)

cursor = db[tweet_collection_name].find({},\
                     {\
                      "retweeted":1, "retweeted_status":1, "is_quote_status":1,\
                      "quoted_status":1,"entities":1, "_id":0, "user":1 \
                      } \
        )

for document in cursor:
    KEY = document["user"]["id"]

    tweeter_name = document["user"]["screen_name"]
    
    if tweeter_name is None:
        print('Skipping', KEY)
        continue
    

    # the list to save his/her mentioned names
    mentioned_name = []
        
    # bulid the mentioned name list
    #for TWEET in VALUE:
    TWEET = document
        # if it is a retweet, only add the connection to the owner of the original tweet
        # not the other ones mentioned in the tweet
    if 'retweeted_status' in TWEET:
        mentioned_name.append({"name":TWEET['retweeted_status']['user']['screen_name'],"id":TWEET['retweeted_status']['user']['id']})
    elif 'quoted_status' in TWEET:
        for MENTIONED in TWEET['entities']['user_mentions']:
            mentioned_name.append({"name":MENTIONED['screen_name'],"id":MENTIONED['id']})
        # if it is a quote, add the owner of the orignial tweet
        try:
            mentioned_name.append({"name":TWEET['quoted_status']['user']['screen_name'],"id":TWEET['quoted_status']['user']['id']})
        except:
            pass
    else:
        for MENTIONED in TWEET['entities']['user_mentions']:
            mentioned_name.append({"name":MENTIONED['screen_name'],"id":MENTIONED['id']})


    # add the connection betweetn the tweeter user and his/her mentioned names
    for name in mentioned_name:
        print(name['name'])
        print(name['id'])
        print(tweeter_name)
        if name['name'].casefold() != tweeter_name.casefold() and is_in_follower_list(name['id']):
            add_connection(tweeter_name.lower(), name['name'].lower())
            


nx.readwrite.gexf.write_gexf(GRAPH, NETWORK_FILE_NAME, encoding='utf-8',
                             version='1.2draft')

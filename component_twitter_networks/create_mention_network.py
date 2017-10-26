import json
import ijson
import networkx as nx
from config import id_or_screen_name

DIRECTORY = 'output/' + id_or_screen_name
FOLLOWERS_FILE_NAME = DIRECTORY + '/followers.json'
FOLLOWER_NAMES_FILE_NAME = DIRECTORY + '/follower_names.json'
TWEETS_FILE_NAME = DIRECTORY + '/followers_tweets.json'
NETWORK_FILE_NAME = DIRECTORY + '/mention_filtered_intranetwork_retweet_and_others_v2.gexf'

TWEETS_COUNT = DIRECTORY + '/followers_tweets_count.json'
MENTIONNAME_COUNT = DIRECTORY + '/followers_mentionname_count.json'

with open(FOLLOWERS_FILE_NAME, encoding='utf-8') as file:
    FOLLOWERS = json.load(file)
with open(FOLLOWER_NAMES_FILE_NAME, encoding='utf-8') as file:
    NAMES = json.load(file)
with open(TWEETS_FILE_NAME, encoding='utf-8') as file:
    # open file with ijson to be able to cope also with larger files
    TWEETS = next(ijson.items(file, ''))

GRAPH = nx.DiGraph()

def add_connection(user_from, user_to):
    if not GRAPH.has_edge(user_from, user_to):
        # if connection does not yet exists, create one
        GRAPH.add_edge(user_from, user_to, weight=0)
    # add +1 to connection weight
    GRAPH[user_from][user_to]['weight'] += 1

def map_id_to_name(user_id):
    if user_id in NAMES:
        return NAMES[user_id]['screen_name']
    print('Name not found for', user_id)
    return None

def is_in_follower_list(screen_name):
    for k in NAMES:
        if NAMES[k]['screen_name'] == screen_name:
            return True
    return False

tw_count = []
name_count = []

# key = user_id, value = n latest tweets of that user
for KEY, VALUE in TWEETS.items():
    tweeter_name = map_id_to_name(KEY)
    if tweeter_name is None:
        print('Skipping', KEY)
        continue

    tw_count.append(len(VALUE))
    mentioned_name = []

    # bulid the mentioned name list
    for TWEET in VALUE:
        # if it is a retweet, only add the connection to the owner of the original tweet
        # not the other ones mentioned in the tweet
        if TWEET['retweeted']:
            mentioned_name.append(TWEET['retweeted_status']['user']['screen_name'])
        elif TWEET['is_quote_status']:
            for MENTIONED in TWEET['entities']['user_mentions']:            
                mentioned_name.append(MENTIONED['screen_name'])
            # if it is a quote, add the owner of the orignial tweet
            try:
                mentioned_name.append(TWEET['quoted_status']['user']['screen_name'])
            except:
                pass
        else:
            for MENTIONED in TWEET['entities']['user_mentions']:            
                mentioned_name.append(MENTIONED['screen_name'])

    name_count.append(len(mentioned_name))

    # add the connection betweetn the tweeter user and his/her mentioned names
    for name in mentioned_name:
        if name.casefold() != tweeter_name.casefold() and is_in_follower_list(name):
            add_connection(tweeter_name.lower(), name.lower())


nx.readwrite.gexf.write_gexf(GRAPH, NETWORK_FILE_NAME, encoding='utf-8',
                             version='1.2draft')


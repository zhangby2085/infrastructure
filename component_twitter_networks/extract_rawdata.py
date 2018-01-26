import json
import os
import codecs
import networkx as nx
from config import id_or_screen_name,project_name,db_host,db_port,db_name,followerID_collection_name,id_name_collection_name,tweet_collection_name,tweet_num
from config import project_target_name, tweet_process_num
from pymongo import MongoClient

PROJECT_DIRECTORY = 'output/project/' + project_name

if not os.path.exists(PROJECT_DIRECTORY):
    os.makedirs(PROJECT_DIRECTORY)

NETWORK_FILE_NAME = PROJECT_DIRECTORY + '/mention_filtered_intranetwork_retweet_and_others_v2.gexf'

TWEETS_COUNT = PROJECT_DIRECTORY + '/followers_tweets_count.json'
MENTIONNAME_COUNT = PROJECT_DIRECTORY + '/followers_mentionname_count.json'


# connect to localhost mongodb
client = MongoClient(db_host, db_port)

# set the database name
db = client[db_name]

# extract the tweets from target user's follower
project_target_name = project_target_name.split(' ')

# find target user's follower IDs
target_followers = []
cursor = db[followerID_collection_name].find({"user":{"$in":project_target_name}})
for doc in cursor:
   target_followers.append(doc["followerID"])

target_followers = list(set(target_followers))
print('There are ',len(target_followers), 'followers for ', ''.join(project_target_name),flush=True)

f_authors = codecs.open(PROJECT_DIRECTORY + '/authors.txt','a','utf-8') 
f_authors_id = codecs.open(PROJECT_DIRECTORY + '/authors_id.txt','a','utf-8') 
f_titles = codecs.open(PROJECT_DIRECTORY + '/titles.txt','a','utf-8') 
f_years = codecs.open(PROJECT_DIRECTORY + '/years.txt','a','utf-8') 
f_venues = codecs.open(PROJECT_DIRECTORY + '/venues.txt' ,'a','utf-8') 

for follower in target_followers:
    #print(follower,flush=True)

    # filter out those ones who has less than 'threshodcount' tweets
    #count = db[tweet_collection_name].find({"user.id":follower}).count()
    #if(count < threshold_num):
    #    continue
    # filter part should be done in data clean not in data collection


    cursor = db[tweet_collection_name].find({"user.id":follower},\
                     {\
                      "text":1, "created_at":1, "retweeted_status.text":1,\
                      "quoted_status.text":1, "_id":0, "user":1 \
                      } \
        ).limit(tweet_process_num)

    
    author = []
    author_id = []
    tweets = []
    time = []
    venue = []

    if cursor.count() ==  0:
        continue

    for doc in cursor:
    
        tweet = doc['text']
        #if tweet == '':
        #    break

        tweet = tweet.replace('\n', ' ')
    
        author.append(doc['user']['screen_name'])
        author_id.append(doc['user']['id_str'])
        time.append(doc['created_at'])

        if doc['user']['location'] == '':
            venue.append('None')
        else:
            venue.append(doc['user']['location'])
    
        if tweet[0:2] == 'RT':
            tweets.append(tweet)
            continue
    
        if 'retweeted_status' in doc:
            tweet = tweet + ' ' + doc['retweeted_status']['text'].replace('\n', ' ')
        if 'quoted_status' in doc:
            tweet = tweet + ' ' + doc['quoted_status']['text'].replace('\n', ' ')
        
        tweets.append(tweet)

    # if the user has no tweets, then skip it
    #if tweet == '':
    #    continue  
    f_authors.write('\n'.join(author) + '\n')
    f_authors_id.write('\n'.join(author_id) + '\n')
    f_titles.write('\n'.join(tweets) + '\n')
    f_years.write('\n'.join(time) + '\n')
    f_venues.write('\n'.join(venue) + '\n')
     
# close files

f_authors.close()
f_authors_id.close()
f_titles.close()
f_years.close()
f_venues.close() 

    
#with open(PROJECT_DIRECTORY + '/authors.txt', 'a', encoding='utf-8') as f: 
#    f.write('\n'.join(author) + '\n')

#with open(PROJECT_DIRECTORY + '/authors_id.txt', 'a', encoding='utf-8') as f: 
#    f.write('\n'.join(author_id) + '\n')
    
#with open(PROJECT_DIRECTORY + '/titles.txt', 'a', encoding='utf-8') as f: 
#    f.write('\n'.join(tweets) + '\n')
    
#with open(PROJECT_DIRECTORY + '/years.txt', 'a', encoding='utf-8') as f: 
#    f.write('\n'.join(time) + '\n')
    
#with open(PROJECT_DIRECTORY + '/booktitles.txt', 'a', encoding='utf-8') as f: 
#    f.write('\n'.join(venue) + '\n')


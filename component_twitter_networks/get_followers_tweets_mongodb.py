import os
import json
import tweepy
from get_tweepy_api import API
from config import id_or_screen_name,project_name,db_name,collection_name
from pymongo import MongoClient

PROJECT_DIRECTORY = 'output/project/' + project_name

id_or_screen_name = id_or_screen_name.split(' ')

Common_Followers = []

for name in id_or_screen_name:
    DIRECTORY = 'output/' + name
    FOLLOWERS_FILE_NAME = DIRECTORY + '/followers.json'

    with open(FOLLOWERS_FILE_NAME, encoding='utf-8') as file:
        FOLLOWERS_USER_IDS = json.load(file)

    Common_Followers.append(FOLLOWERS_USER_IDS)

flat_list = [item for sublist in Common_Followers for item in sublist]

Common_Followers = set(flat_list)
Common_Followers = list(Common_Followers)

LOG_FILE_NAME = PROJECT_DIRECTORY + '/followers_tweets_mongodb_log.json'


# iterate over followers
# get 100 latest tweets from each of those

TWEETS = {}
LOG = []

# connect to localhost mongodb
client = MongoClient('localhost', 27017)

# set the database name
#db = client.TwitterFollowersTUT
#db = client.testTweets
db = client[db_name]

for follower_id in Common_Followers:
    TWEETS[follower_id] = []
    cursor = tweepy.Cursor(API.user_timeline, user_id=follower_id, count=500).items(500)
    try:
        for item in cursor:
            #TWEETS[follower_id].append(item._json)
            # insert into the collection tweets
            #result = db.tweets.insert_one(item._json)
            result = db[collection_name].insert_one(item._json)

    except tweepy.TweepError as error:
        ERROR_STRING = 'Tweepy error: {}. Failed to retrieve tweets of follower {}'.format(error.reason, follower_id)
        print(ERROR_STRING)
        # LOG[follower_id] = ERROR_STRING
        LOG.append(ERROR_STRING)
        continue
    except StopIteration:
        print("StopIteration")
        break
    except:
        print("Unkown error")
        LOG.append('Unknown error while processing follower {}'.format(follower_id))

#with open(TWEETS_FILE_NAME, 'w') as f:
#    json.dump(TWEETS, f, indent=4)

print("save the log")
if not os.path.exists(PROJECT_DIRECTORY):
    os.makedirs(PROJECT_DIRECTORY)
with open(LOG_FILE_NAME, 'w') as f:
    json.dump(LOG, f, indent=4)

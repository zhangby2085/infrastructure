import os
import json
import tweepy
from get_tweepy_api import API
from config import id_or_screen_name


id_or_screen_name = id_or_screen_name.split(' ')

for name in id_or_screen_name:
    DIRECTORY = 'output/' + name
    if not os.path.exists(DIRECTORY):
        os.makedirs(DIRECTORY)
    FOLLOWERS_FILE_NAME = DIRECTORY + '/followers.json'

    FOLLOWERS = []
    FOLLOWERS_CURSOR = tweepy.Cursor(API.followers_ids, id=name, count=5000).items()

    # get target user's followers
    while True:
        try:
            FOLLOWER = FOLLOWERS_CURSOR.next()
            FOLLOWERS.append(FOLLOWER)
        except StopIteration:
            break

    print('User had', len(FOLLOWERS), 'followers')

    with open(FOLLOWERS_FILE_NAME, 'w') as f:
        json.dump(FOLLOWERS, f, indent=4)


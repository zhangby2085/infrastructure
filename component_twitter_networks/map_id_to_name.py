import json
from get_tweepy_api import JSON_API
from config import id_or_screen_name

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

for name in id_or_screen_name:
    DIRECTORY = 'output/' + name
    FOLLOWERS_FILE_NAME = DIRECTORY + '/followers.json'
    FOLLOWER_NAMES_FILE_NAME = DIRECTORY + '/follower_names.json'

    with open(FOLLOWERS_FILE_NAME, encoding='utf-8') as file:
        USER_IDS = json.load(file)


    USERS = {}

    for chunk_of_ids in chunks(USER_IDS, 100):
        USERS_LIST = JSON_API.lookup_users(user_ids=chunk_of_ids)
        for user in USERS_LIST:
            USERS[user['id']] = user

    with open(FOLLOWER_NAMES_FILE_NAME, 'w') as f:
        json.dump(USERS, f, indent=4)

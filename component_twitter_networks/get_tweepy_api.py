import tweepy
from twitter_api_tokens import CONSUMER_KEY, CONSUMER_SECRET, OAUTH_TOKEN, OAUTH_TOKEN_SECRET

AUTH = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
AUTH.set_access_token(OAUTH_TOKEN, OAUTH_TOKEN_SECRET)

API = tweepy.API(AUTH, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
JSON_API = tweepy.API(
    AUTH,
    wait_on_rate_limit=True,
    wait_on_rate_limit_notify=True,
    parser=tweepy.parsers.JSONParser()
)
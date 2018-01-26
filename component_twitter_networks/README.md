# ABOUT
This repository contains the code to collect the followers and followers's tweets of the target(s) account. Then performan a people recommendation for a given user who is one of the followers.  
The tasks include:
1. collect followers and their tweets, save into MongoDB
2. Language detection for each tweet
3. Identity classification for each follower ( person or organisation)
4. Create mention network (not following network) for Gephi network analysis
5. Perform people recommendation using basic TF-IDF features on English-only tweets

# Environment
Dev and test on:  
OS: OSX Version 10.12.6  
Python:  
- Python 3.5.4
- Python 2.7.13 ( Only used by Humanizer for follower identity classification)

conda: 4.3.21  
langdetect: 1.0.7  
Humanizr: https://github.com/networkdynamics/humanizr  

# HOW TO
### Collect Data
Install and run MongoDB in background
https://docs.mongodb.com/manual/tutorial/install-mongodb-on-os-x/
First get the followers ID
```python
python get_followers.py
```
Collect followers's tweets and save into MongoDB
```python
python get_followers_tweets_mongodb.py
```
### Data Clean & Preprocess 
Extract the text, author, venue and date of all the tweets
```sh
python extract_rawdata.py
```
Create the Gephi file of mention network
```sh
Python create_mention_network.py
```
Generate the txt file of the mention network in which contains the author, people he/she mentioned and the number of mentions
```sh
./create_mention_network.sh
```
Clean the tweet text file by removing the special character ^M
```sh
./remove_M.sh 
```
Further clean the tweets ( removing emoji, digital, special character, etc ) and detect the language of each tweet ( need package langdetect: conda install -c conda-forge langdetect ).
Classify the followers into person and organisations ( need tool Humanizr ).
```sh
./save_data_local_copy.sh
./classify_organizations.sh -o OUTPUT_DIR/per_org.txt  PROJECT_DIR/data_by_id/
```
Extract final target data by defining the language and the least number of tweets
```sh
./extract_target_data.sh
```
### Run the recommendation
Run the recommendation system for a given user and output the result as a JSON file

```python
from recommendationsys import recommendationsys
remd = recommendationsys('Oct 1 2017')
user = "jnkka"
result = remd.recommendationV4(user,3)

for author in result['recommendations']:
    print(author['name']  + ' ' + str(author['relevancy']))

remd.save_json('jnkka')
```

# Source Code Structure
The source code direcotry structure is as following ( with the project_name = 'Tampere3_v5' in config.py ):
```sh
.
|____clean_text.py
|____clean_tweets.py
|____config.py
|____create_mention_network.py
|____create_mention_network_recsys.sh
|____extract_rawdata.py
|____extract_rawdata_multithread.py
|____extract_target_data.sh
|____get_followers.py
|____get_followers_tweets_mongodb.py
|____get_tweepy_api.py
|____output
| |____project
| | |____Tampere3_v5
| | | |____authors.txt
| | | |____authors_id.txt
| | | |____titles.txt
| | | |____venues.txt
| | | |____years.txt
| | | |____data_by_id
| | | | |____100051597.json
| | | | |____100127207.json
...
| | | | |____628180839.json
|____recommendationsys.py
|____remove_M.sh
|____save_data_local_copy.sh
|____twitter_api_tokens.py
```

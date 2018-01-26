# user id or screen name of the target user 
# from who the follower and their tweets will be collected
#id_or_screen_name = 'winterdamsel Roc_Yao zhangby2085'
id_or_screen_name = 'TampereUniTech TAMK_UAS UniTampere Tampere_3'

# project name
project_name = 'Tampere3_v5'

# the scope of the project, should be the same of the id_or_screen_name
# or a sub set of it
project_target_name = 'TampereUniTech'

# mongoDB host
db_host = 'localhost'

# mongoDB port
db_port = 27017

# database name 
db_name = 'tampere3'

# tweet collection name
tweet_collection_name = 'tweets'

# follwers id collection name
followerID_collection_name = 'followerID'

# log collection name
log_collection_name = 'log'

# followers id:name mapping collection name
id_name_collection_name = 'id_name'

# the most number of tweets to collect for each follower
tweet_num = 300

# the least number of tweets to process
tweet_process_num = 5

# this setting is used in data clean phase to filter out those who have 
# less tweets than this threshold
threshold_num = '1'

# set the target language
lan = 'en'

from clean_text import clean_text
import codecs
from config import project_name

PROJECT_DIRECTORY = 'output/project/' + project_name

INPUT_FILE = PROJECT_DIRECTORY + '/titlesLF.txt'
OUTPUT_FILE_TWEET = PROJECT_DIRECTORY + '/cleantitles.txt'
OUTPUT_FILE_LAN = PROJECT_DIRECTORY + '/lan.txt'


input_file = codecs.open(INPUT_FILE,'r','utf-8')
input_file_tweet = codecs.open(OUTPUT_FILE_TWEET,'w','utf-8')
input_file_lan = codecs.open(OUTPUT_FILE_LAN,'w','utf-8')

for i in input_file:
    newtweet, lan = clean_text(i)
    input_file_tweet.write(newtweet+'\n')
    input_file_lan.write(lan+'\n')

input_file.close()
input_file_tweet.close()
input_file_lan.close()

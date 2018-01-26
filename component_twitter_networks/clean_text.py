from langdetect import detect
import re
def clean_text(text):
    
    # remove the 'RT ' at the beginning if it is a retweet
    if text[0:3] == 'RT ':
        text = text[3:]
    
    text = text.lower()
        
    # this is for USC-2
    # remove emojis
    myre = re.compile(u'('
        u'\ud83c[\udf00-\udfff]|'
        u'\ud83d[\udc00-\ude4f\ude80-\udeff]|'
        u'\ufe0f|'
        u'[\u2600-\u26FF\u2700-\u27BF]|'
        '@\S*\s?|#|'   # remove @ mention names and hastag sign
        'http[s]?://\S*\s' # remove url
        ')+', 
        re.UNICODE)
    
    newtext = myre.sub(' ',text)
    
    newtextlist = newtext.split()
    
    # remove punctuations and then count how many non-english alph words 
    mypunctuation = '!"$%&()*+,./:;<=>?@[\\]^`{|}~'
    
    newtext = []
    
    for t in newtextlist:
        # remove the punctuations first
        t = t.strip(mypunctuation)      
        newtext.append(t)   

    newtext = ' '.join(newtext)
 
    try:   
        lan = detect(newtext)
    except:
        lan = 'none'
    
    return (newtext,lan)

# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 08:37:16 2017

@author: secoder
"""

#!/usr/bin/env python
"""
Minimal Example
===============
Generating a square wordcloud from the US constitution using default arguments.
"""
from wordcloud import WordCloud

#d = path.dirname(__file__)

# Read the whole text.
#text = open(path.join(d, 'HICSS_titles.txt')).read()

def wordcloudit(text):

    # Generate a word cloud image
    wordcloud = WordCloud().generate(text)

    # Display the generated image:
    # the matplotlib way:
    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(wordcloud)
    plt.axis("off")

    # lower max_font_size
    #wordcloud = WordCloud(max_font_size=40).generate(text)
    #plt.figure()
    #plt.imshow(wordcloud)
    #plt.axis("off")
    #plt.show()

# The pil way (if you don't have matplotlib)
#image = wordcloud.to_image()
#image.show()
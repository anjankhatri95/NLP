
import nltk
import string
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.chunk import ne_chunk

import matplotlib.pyplot as plt

#opening a file
file=open("COVIDfakeNews.txt","r",encoding='utf-8')
read= file.read()
#lower case
all_words= read.lower()
#tokenize the words from the file
tokenize_word= word_tokenize(all_words)

#print(tokenize_word)

#filtering and eliminating the useless words.
string.punctuation
uselesstoken= nltk.corpus.stopwords.words("english")+list(string.punctuation)

filtered_words = [w for w in tokenize_word if not w in uselesstoken]

#print(filtered_words)

#parts of speech tagging
word_tags= nltk.pos_tag(filtered_words)

#chunking the word with Name entities
chunking_word= ne_chunk(word_tags)
#print(chunking_word)

#calculating the frequecny of the words used
word_counter= Counter(word_tags)
most_used_word=word_counter.most_common()
print(most_used_word)

#plot
sorted_word_count= sorted(list(word_counter.values()), reverse=True)
#print(sorted_word_count)

plt.loglog(sorted_word_count)
plt.ylabel("Frequency")
plt.xlabel("Words index")
plt.show()

#sentiment analysis
import textblob
from textblob import TextBlob
Text_analysis = TextBlob(all_words)
text_sentiments= Text_analysis.sentiment.polarity
print("polarity of text",text_sentiments)

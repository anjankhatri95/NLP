import nltk
import textblob
from mediawiki import MediaWiki
from textblob import TextBlob
wikipedia = MediaWiki()
 #assigning wikipedia page
politics = wikipedia.page('2020 United States presidential election')
history= wikipedia.page('Albigensian Crusade')
science= wikipedia.page('Black hole')
celebrity=wikipedia.page('Dwayne Johnson')
movie=wikipedia.page('Thor: Ragnarok')

#texting postivie and negative text
positive_text="The product is awesome"
negative_text="The product is useless"

political_content= politics.content
history_content=history.content
science_content=science.content
celebrity_content=celebrity.content
movie_content=movie.content

#textblob analysis
politics_tb=TextBlob(political_content)
history_tb=TextBlob(history_content)
science_tb=TextBlob(science_content)
celebrity_tb=TextBlob(celebrity_content)
movie_tb= TextBlob(movie_content)

#text blob on postive and negative text
pos_text_tb= TextBlob(positive_text)
neg_text_tb= TextBlob(negative_text)

#sentiment polarity
political_sentiments=politics_tb.sentiment.polarity
history_sentiments=history_tb.sentiment.polarity
science_sentiments=science_tb.sentiment.polarity
celebrity_sentiments=celebrity_tb.sentiment.polarity
movie_sentiments=movie_tb.sentiment.polarity

#sentiments on positive and negative text
pos_text_sentiments= pos_text_tb.sentiment.polarity
neg_text_sentiments= neg_text_tb.sentiment.polarity

#printing polarity
print("political", political_sentiments)
print("History",history_sentiments)
print("science",science_sentiments)
print("celebrity",celebrity_sentiments)
print("Movie",movie_sentiments)

#printing pos and neg text
print("Postive text", pos_text_sentiments)
print("Negative text",neg_text_sentiments)
#for the positive sentence polarity is 1 and negative text the polarity is -0.5 close of -1.
#it think it does better.
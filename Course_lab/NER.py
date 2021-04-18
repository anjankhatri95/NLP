import nltk

from nltk.chunk import ne_chunk
from nltk.tokenize import word_tokenize
from mediawiki import MediaWiki
wikipedia = MediaWiki()

p = wikipedia.page('Free Guy')
content= p.content

#word tokenize
wordstoken= word_tokenize(content)


#pos tagging
wordstag= nltk.pos_tag(wordstoken)

#Chunking
chunk= nltk.ne_chunk(wordstag)

#output

#print(wordstoken)
#print(wordstag)
print(chunk)

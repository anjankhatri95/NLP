import nltk

from nltk.corpus import brown
brown_tagged_sents = brown.tagged_sents(categories='news')

size = int(len(brown_tagged_sents) * 0.9)

train_sents = brown_tagged_sents[:size]
test_sents = brown_tagged_sents[size:]

#train the tagger
unigram_tagger = nltk.UnigramTagger(train_sents)

#calculate the accuracy
print("Results on train set {0}".format(unigram_tagger.evaluate(train_sents)))
print("Results on test set {0}".format(unigram_tagger.evaluate(test_sents)))

def_tagger= nltk.DefaultTagger("NN")
uni_tagger=  nltk.UnigramTagger(train_sents, backoff=def_tagger)

print("Results on train set {0}".format(uni_tagger.evaluate(train_sents)))
print("Results on test set {0}".format(uni_tagger.evaluate(test_sents)))

#bigram

bigram_tagger = nltk.BigramTagger(train_sents)

#calculate the accuracy
print("Results on train set (bigram) {0}".format(bigram_tagger.evaluate(train_sents)))
print("Results on test set (bigram){0}".format(bigram_tagger.evaluate(test_sents)))

#using unigram tagger as a backoff.
bi_tagger=  nltk.UnigramTagger(train_sents, backoff=unigram_tagger)

print("Results on train set (bigram) {0}".format(uni_tagger.evaluate(train_sents)))
print("Results on test set (bigram) {0}".format(uni_tagger.evaluate(test_sents)))

#5 Repeat #4 with a TrigramTagger using a Bigramtagger as backoff

#trigram
#train the tagger
Trigram_tagger = nltk.TrigramTagger(train_sents)

#calculate the accuracy
print("Results on train set (trigram) {0}".format(Trigram_tagger.evaluate(train_sents)))
print("Results on test set (trigram){0}".format(Trigram_tagger.evaluate(test_sents)))

#using biGramtragger as a backoff
bi_tagger=  nltk.BigramTagger(train_sents, backoff=bigram_tagger)

print("Results on train set (trigram) {0}".format(bi_tagger.evaluate(train_sents)))
print("Results on test set (trigram) {0}".format(bi_tagger.evaluate(test_sents)))


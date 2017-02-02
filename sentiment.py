# at a high level, sentimental analysis is "opinion mining" that identify the polarity
# of the text at document level, sentence level, feature level

# 1. Knowledge Method - the first approach is assigning the value of the word directyl (happy, sad, angry)
# 2. Statistical methods - Machine Learning Algorithms (Support Vector Machines, Latent Semantic Analysis, Naive Bayes, deep/shallow parsing)
# 3. Hybrid Approach (Combination of Knowledge representation and statistical model)

# They would not let my dog stay in this hotel VS I would not let my dog stay in this hotel
# can you use this in production? there is a boundary between research and production. there are lot of tradeoffs

import nltk
import pandas as pd
import numpy as np
import scipy

tweets = []

pos_tweets = [('I love this car', 'positive'),
              ('This view is amazing', 'positive'),
              ('I feel great this morning', 'positive'),
              ('I am so excited about the concert', 'positive'),
              ('He is my best friend', 'positive')]

neg_tweets = [('I do not like this car', 'negative'),
              ('This view is horrible', 'negative'),
              ('I feel tired this morning', 'negative'),
              ('I am not looking forward to the concert', 'negative'),
              ('He is my enemy', 'negative'),
              ('She is sad', 'negative'),]

test_tweets = [
	    (['feel', 'happy', 'this', 'morning'], 'positive'),
	    (['larry', 'friend'], 'positive'),
	    (['not', 'like', 'that', 'man'], 'negative'),
	    (['house', 'not', 'great'], 'negative'),
	    (['your', 'song', 'annoying'], 'negative')]

# put positive and negative tweets in the same array, split by word
for (words, sentiment) in pos_tweets + neg_tweets:
    words_filtered = [e.lower() for e in words.split() if len(e) >= 3]
    tweets.append((words_filtered, sentiment))

# create a feature list
def get_words_in_tweets(tweets):
    all_words = []
    for (words, sentiment) in tweets:
      all_words.extend(words)
    return all_words

def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    word_features = wordlist.keys()
    return word_features

word_features = get_word_features(get_words_in_tweets(tweets))


def extract_features(document):
	    document_words = set(document)
	    features = {}
	    for word in word_features:
	        features['contains(%s)' % word] = (word in document_words)
	    return features

# training algorithm here
training_set = nltk.classify.apply_features(extract_features, tweets)
classifier = nltk.NaiveBayesClassifier.train(training_set)

print(classifier.show_most_informative_features(32))

new_tweet = "Benjamin is sad"
print(classifier.classify(extract_features(new_tweet.split())))


# NOT SURE IF THIS IS NEEDED
# def train(labeled_featuresets, estimator=ELEProbDist):
# 	    # ...
# 	    # Create the P(label) distribution
# 	    label_probdist = estimator(label_freqdist)
# 	    # ...
# 	    # Create the P(fval|label, fname) distribution
# 	    feature_probdist = {}
# 	    # ...
# 	    return NaiveBayesClassifier(label_probdist, feature_probdist)

import csv
import os
import re
from string import punctuation
from nltk import TweetTokenizer
from nltk.corpus import wordnet
import jsonlines
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

from src.VariablesList import NER_SENTIMENT_FILES, EMOTICONS_LEXICON_FILE, TWEETS_FILE, NER_OUTPUT_FILE, EMOTIONS
from src.contractions import expandContractions


def readFile(file):
    """ Returns the dictionary (word, score) pair for the emotion from the given lexicon file """
    d = {}
    with open(file, 'r') as f:
        for line in f:
            (key, val) = line.strip().split("\t")
            d[key] = float(val)
    return d


def createEmojiVector():
    """" Create Emoji vector for each emotions"""
    emoji_vector = {}
    with open(EMOTICONS_LEXICON_FILE, 'r') as file:
        for line in file:
            curr = line.strip().split(",")
            emoji = curr[1]
            value = [float(x) for x in curr[5:]]
            emoji_vector[emoji] = value
    return emoji_vector


def create_dictionaries():
    """" Create a list of the lexicon file by sorting the lexicon emotion file and returns it """
    files = os.listdir(NER_SENTIMENT_FILES)
    lexicon_dictionary = []
    files.sort()
    for file in files:
        if not file.startswith("."):
            if os.path.isfile(os.path.join(NER_SENTIMENT_FILES, file)):
                lexicon_dictionary.append(readFile(os.path.join(NER_SENTIMENT_FILES, file)))
    return lexicon_dictionary


def get_wordnet_pos(word):
    """Map POS tag to to perform lemmatization"""
    tag = pos_tag([word])[0][1][0].upper()
    tag_dict = {
        "N": wordnet.NOUN,
        "V": wordnet.VERB,
        "J": wordnet.ADJ,
        "R": wordnet.ADV
    }
    return tag_dict.get(tag, wordnet.NOUN)


def processTweet(tweet):
    _stopwords = set(stopwords.words('english') + list(punctuation) + ['AT_USER', 'URL', 'RT', 'rt', 'at_user', 'url'])
    tweet = tweet.lower()  # convert text to lower-case
    tweet = expandContractions(tweet)  # expand the contractions to remove the stop words
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', tweet)  # remove URLs
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)  # remove the # in #hashtag
    tokenizer = TweetTokenizer(strip_handles=True, reduce_len=True)  # tokenize the tweet and remove the handle
    tweet = tokenizer.tokenize(tweet)
    # remove the stop words from the tokenzied tweet
    tweet = [word for word in tweet if word not in _stopwords]
    # perform lemmatization on the words which helps to find the root of the word
    # The Lemmatization will not be performed for NER as the Lexicons are derived from Twitter
    # tweet = [WordNetLemmatizer().lemmatize(w, get_wordnet_pos(w)) for w in tweet]
    return tweet


def label_tweets_using_NER():
    # create dictionaries for the lexicons
    lexical = create_dictionaries()
    emojis_vector = createEmojiVector()
    # variable to keep the count of the processed tweet
    tweet_count = 0
    with open(NER_OUTPUT_FILE, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["Id", "Date", "Tweet", "Emotion"])
    # open the tweet file to process them
    with jsonlines.open(TWEETS_FILE, 'r') as reader:
        for obj in reader:
            if tweet_count and tweet_count % 10000 == 0:
                print("tweet_count", tweet_count)
            # Only process the tweets which has location, not truncated, language is English and Tweet location is US
            if obj["lang"] == "en" and not obj["truncated"] and obj["place"] is not None and obj["place"][
                "country_code"] == "US":
                tweet_count += 1
                # get the retweet text based on if not truncated text as this will give the maximum length for the tweet
                if "retweeted_status" in obj:
                    if obj["retweeted_status"]["truncated"]:
                        continue
                    if "full_text" in obj["retweeted_status"]:
                        tweet = obj["retweeted_status"]["full_text"]
                    else:
                        tweet = obj["retweeted_status"]["text"]
                elif 'full_text' in obj:
                    tweet = obj['full_text']
                else:
                    tweet = obj['text']
                tweet = tweet.replace("\n", " ")
                # List to capture the emotion in eight different categories
                tweet_score = [0.0] * len(EMOTIONS)
                tweet_score = np.asarray(tweet_score)
                # process the tokenized tweet
                for word in processTweet(tweet):
                    for i in range(len(lexical)):
                        if word in lexical[i]:
                            tweet_score[i] += lexical[i][word]
                    # if word in emojis_vector:
                    #     tweet_score += np.asarray(emojis_vector[word])
                # to write into csv
                result = [obj["id"], obj["created_at"], tweet]
                if np.max(tweet_score) == 0:
                    result.append("neutral")
                else:
                    result.append(EMOTIONS[int(np.argmax(tweet_score))])
                # writing to csv file
                with open(NER_OUTPUT_FILE, 'a', newline='', encoding='utf-8') as csvfile:
                    # creating a csv writer object
                    csvwriter = csv.writer(csvfile)
                    # writing the fields
                    csvwriter.writerow(result)
    print("total processed:", tweet_count)



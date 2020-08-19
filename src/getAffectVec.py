import csv
import jsonlines
import numpy as np
from src.ReadLexicons import processTweet
from src.VariablesList import AFFECT_VEC_FILE, AFFECT_VEC_OUTPUT_FILE, TWEETS_FILE


def readEmbeddingFile(file):
    embedding = {}
    with open(file, 'r') as handle:
        for line in handle:
            row = line.strip().split(" ")
            word = row[0]
            value = [float(x) for x in row[1:]]
            embedding[word] = value
    return embedding


embeddings = readEmbeddingFile(AFFECT_VEC_FILE)
not_found = [0] * 239
range_of_emotion = "levity hate loyalty melancholy anxiety embarrassment regard stress gusto compunction cynicism situation anger umbrage favor meekness compassion withdrawal scare unrest calm courage despair fidget shyness apathy hysteria shadow resentment optimism heartstrings bonheur dudgeon merriment hope foreboding envy interest relaxed cruelty surprise helplessness trust solicitude satisfaction suspense fondness dolor weakness electricity esteem woe relieved wonder attachment pessimism malice love compatibility timidity blessedness exultation tumult alienation humility powerlessness complacency gloom aggression sensation antipathy gloat doubt empathy consciousness ingratitude hopelessness signal alarm dislike stir distance smugness repentance easiness friendliness gravity displeasure discouragement pique benevolence chagrin tension togetherness panic eagerness pleasure excitement mood animosity defeatism worship repugnance grudge euphoria antagonism trait brotherhood stewing pity daze sympathy annoyance encouragement buoyancy disgust devotion triumph contempt belonging sinking fear unhappiness trepidation admiration disapproval indifference affection astonishment oppression languor coolness liking behaviour peace misogyny bang cheerfulness creeps agitation boredom gratification hurt agape concern ardor mourning harassment contentment closeness surprised confusion presage approval state wrath dander reverence content amusement indignation fearlessness depreciation expectation tenderness misery depression forgiveness willies fit comfort shame apprehension delight joy jealousy aggravation chill warpath serene anticipation exuberance resignation gratitude despondency nirvana lividity emotion disappointment horror grief weight distress intoxication irritation insecurity pride fever rejoicing impatience politeness tranquillity hilarity fury gladness sadness thing nausea calmness fulfillment ecstasy elation playfulness exhilaration titillation gratefulness diffidence radiance sorrow confidence security ego hostility frustration attrition angst shock happiness preference enthusiasm isolation conscience scruple worry earnestness malevolence awe guilt identification".split(
    " ")
# emotions_list = ["anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust", "neutral"]
# emotion_index = tuple([12, 180, 115, 121, 174, 205, 40, 42])


def getTweetEmbedding(tweet):
    token_tweet = processTweet(tweet)
    embedding = [embeddings[token] if token in embeddings.keys() else not_found for token in token_tweet]
    embedding = np.asarray(embedding)
    if len(token_tweet) == 0:
        return np.asarray(not_found)
    else:
        return np.mean(embedding, axis=0)


def label_tweets_using_affect():
    # variable to keep the count of the processed tweet
    tweet_count = 0
    with open(AFFECT_VEC_OUTPUT_FILE, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["Id", "Date", "Tweet", "Emotion", "Location"])

    # open the tweet file to process them
    with jsonlines.open(TWEETS_FILE, 'r') as reader:
        for obj in reader:
            if obj["lang"] == "en" and not obj["truncated"] and obj["place"] is not None and obj["place"][
                "country_code"] == "US":
                # get the retweet text based on if not truncated text as this will give the maximum length for the tweet
                if "retweeted_status" in obj:
                    if obj["retweeted_status"]["truncated"]:
                        break
                    if "full_text" in obj["retweeted_status"]:
                        tweet = obj["retweeted_status"]["full_text"]
                    else:
                        tweet = obj["retweeted_status"]["text"]
                elif 'full_text' in obj:
                    tweet = obj['full_text']
                else:
                    tweet = obj['text']
                tweet_emotions = getTweetEmbedding(tweet)
                # concerned_emotion = (
                #     tweet_emotions[12], tweet_emotions[180], tweet_emotions[115], tweet_emotions[121],
                #     tweet_emotions[174],
                #     tweet_emotions[205], tweet_emotions[40], tweet_emotions[42])

                concerned_emotion = (
                    tweet_emotions[12], tweet_emotions[180],
                    tweet_emotions[174],
                    tweet_emotions[205])

                # to write into csv
                result = [obj["id"], obj["created_at"], tweet]
                if np.max(tweet_emotions) <= 0:
                    result.append("neutral")
                else:
                    result.append(range_of_emotion[int(np.argmax(tweet_emotions))])
                if obj["place"] is not None:
                    result.append(obj["place"]["full_name"].split(","))
                else:
                    result.append("None")
                tweet_count += 1
                # writing to csv file
                with open(AFFECT_VEC_OUTPUT_FILE, 'a', newline='', encoding='utf-8') as csvfile:
                    # creating a csv writer object
                    csvwriter = csv.writer(csvfile)
                    # writing the fields
                    csvwriter.writerow(result)


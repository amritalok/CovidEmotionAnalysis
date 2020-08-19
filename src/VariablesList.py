# INPUT FILES
TWEETS_FILE = "/Users/amritalok/PycharmProjects/CovidEmotionAnalysis/data/tweets.jsonl"
NER_SENTIMENT_FILES = "/Users/amritalok/PycharmProjects/CovidEmotionAnalysis/lexicons/OneFilePerEmotion"
EMOTICONS_LEXICON_FILE = "/Users/amritalok/PycharmProjects/CovidEmotionAnalysis/lexicons/afjs-weiAvg-620-emoji.csv"
AFFECT_VEC_FILE = "/Users/amritalok/PycharmProjects/CovidEmotionAnalysis/lexicons/AffectVec-v1.0/AffectVec-data.txt"

# OUTPUT FILES
NER_OUTPUT_FILE = "/Users/amritalok/PycharmProjects/CovidEmotionAnalysis/output/NER_TWEETS_OUTPUT.csv"
AFFECT_VEC_OUTPUT_FILE = "/Users/amritalok/PycharmProjects/CovidEmotionAnalysis/output/AFFECTVEC_TWEETS_OUTPUT.csv"

# Emotions to Analyze
EMOTIONS = {0: "anger", 1: "anticipation", 2: "disgust", 3: "fear", 4: "joy", 5: "sadness", 6: "surprise", 7: "trust"}
# EMOTIONS = {0: "anger", 3: "fear", 4: "joy", 5: "sadness", 7: "trust"}


EMOTIONS_TO_PLOT = {"anger": "#972E2F", "anticipation": "#CB8655", "disgust": "#AF436B", "fear": "#568A3E",
                    "joy": "#B95F31", "sadness": "#6B6DA9", "surprise": "#7AB960", "trust": "blue", "neutral": "black"}

# EMOTIONS_TO_PLOT = {"anger": "#972E2F", "fear": "#568A3E", "joy": "#B95F31", "sadness": "#6B6DA9", "neutral": "black"}

import pandas as pd, numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
import seaborn as sns
import matplotlib.dates as mdates

from src.ReadLexicons import label_tweets_using_NER
from src.VariablesList import EMOTIONS_TO_PLOT, NER_OUTPUT_FILE, AFFECT_VEC_OUTPUT_FILE
from src.getAffectVec import label_tweets_using_affect

pd.set_option('display.max_rows', None)


def draw_inference(ProcessedLexiconFile):
    # read csv
    data = pd.read_csv(ProcessedLexiconFile)
    # create a seaborn object
    sns.set(context='poster', style='darkgrid', rc={'figure.figsize': (45, 40)})
    # context = 'poster', style = 'darkgrid', palette = 'deep', font_scale = 1,rc={'figure.figsize': (40, 35)}
    # sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})

    # caluclate the overall statistics
    emotions = data["Emotion"]
    count = emotions.value_counts()
    percent = emotions.value_counts(normalize=True)
    percent100 = percent.mul(100).round(1).astype(str) + '%'
    print(pd.DataFrame({"count": count, "per": percent, "percentage": percent100}))
    # process the string and capture only the date
    data["just_date"] = pd.to_datetime(data['Date']).dt.normalize()
    # mask = data["just_date"] >= "2020-01-20"
    # data = data[mask]
    # create a new Dataframe with only emotion and date
    data_emotion = pd.DataFrame({"date": data["just_date"], "emotion": data["Emotion"]})
    # mask = data_emotion_temp["date"] >= "2020-01-20"
    # data_emotion = data_emotion_temp[mask]
    # print(data_emotion)

    # Aggregate the count of tweets for each emotion per day
    time_series = data_emotion.groupby(['date', 'emotion'])['emotion'].agg(
        lambda x: x.value_counts())  # .reset_index(name='count')
    time_series_percentage = time_series.groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).reset_index(
        name='count')

    # create a subplot
    fig, axes = plt.subplots(len(EMOTIONS_TO_PLOT), 1, sharex=False, sharey=True)
    custom_ylim = (0, 75)
    plt.setp(axes, ylim=custom_ylim, yticks=[10, 20, 30, 40, 50, 60, 70])
    plt.tight_layout()
    # loop through each emotion
    for emotion, ax in zip(EMOTIONS_TO_PLOT.keys(), axes.flat):
        # Filter all rows for the current emotion -> gives a boolean object
        data = time_series_percentage["emotion"] == emotion
        # Create a new dataframe to get the rows for the current emotion
        emotion_timeseries = time_series_percentage[data]
        # Set x-axis major ticks to weekly interval, on Mondays
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MONDAY))
        # Format x-tick labels as 3-letter month name and day number
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        # plot the data
        emotion_timeseries.plot.area(x="date", y="count", linewidth=0.0, label=emotion, ax=ax,
                                     color=EMOTIONS_TO_PLOT[emotion])
    plt.legend()
    plt.show()


if __name__ == "__main__":
    inference_type = int(input("Enter 1 for NER 2 for AffectVec: "))

    if inference_type == 1:
        label_tweets_using_NER()
        draw_inference(NER_OUTPUT_FILE)
    elif inference_type == 2:
        label_tweets_using_affect()
        draw_inference(AFFECT_VEC_OUTPUT_FILE)
    else:
        print("Incorrect input, Try Again!")

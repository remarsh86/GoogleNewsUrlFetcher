import pandas as pd
import os
import config
import matplotlib.pyplot as plt


def create_scatter_plots(topic_data, features):
    # todo: Add regression line
    for feature in features:
        # Plot values
        ax = topic_data.plot(x='rank', y=feature, kind='scatter')
        ax.set_xlim(1, 100)
        plt.xticks(rotation=90)
        plt.show()


def feature_comparison(topic_data, features):
    for feature in features:
        # Find mean for each value
        print(f'Mean of {feature} in {topic}: {topic_data[feature].mean(skipna=True)}')
        # todo: add to dataframe of mean values


if __name__ == '__main__':
    data = pd.read_csv(os.path.join(config.get_app_root(), "evaluation", 'politics_topics.csv'))

    features = ['readability', 'sentence_level_sentiment', 'sentence_level_objectivity', 'bias', 'credibility',
                'trust_metric', 'google_page_rank', 'alexa_reach_rank']

    # todo: create dataframe (if dataframe is mutable object type), and add mean values per feature per topic


    for topic in data.topic.unique():
        topic_data = data[data['topic'] == topic]
        # create_scatter_plots(topic_data, features)
        feature_comparison(topic_data, features)



    # fig, ax = plt.subplots()
    # data.groupby('topic').plot(x='rank', y='sentence_level_sentiment', ax=ax, legend=False)
    # plt.show()

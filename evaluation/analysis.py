import pandas as pd
import os
import config
import matplotlib.pyplot as plt
import numpy as np


def create_scatter_plots(topic_data, features):
    # todo: Add regression line
    # todo: Add title for topic_data
    for feature in features:
        ax = topic_data.plot(x=feature, y='rank', kind='scatter')
        ax.set_xlim(1, 101)
        ax.set_ylim(1, 101)
        plt.xticks(rotation=90)
        plt.show()

        # ax = topic_data.plot(x='rank', y=feature, kind='scatter')
        # ax.set_xlim(1, 101)
        # ax.set_ylim(1, 101)
        # plt.xticks(rotation=90)
        # plt.show()


def get_output_averages(topic_dataframe, feature, df):
    """

    :param topic_dataframe:
    :param feature:
    :param df:
    :return:
    """
    topic_dataframe.reset_index(drop=True, inplace=True)
    for INL in features:
        top_1 = topic_dataframe[INL][0]
        print(f'Input bias of {INL} in {topic}: {topic_dataframe[INL].mean(skipna=True)}')
        input_bias = topic_dataframe[INL].mean(skipna=True)
        top_3 = topic_dataframe[INL][topic_dataframe['rank'] <= 3].mean(skipna=True)
        top_5 = topic_dataframe[INL][topic_dataframe['rank'] <= 5].mean(skipna=True)
        top_10 = topic_dataframe[INL][topic_dataframe['rank'] <= 10].mean(skipna=True)
        top_20 = topic_dataframe[INL][topic_dataframe['rank'] <= 20].mean(skipna=True)
        top_50 = topic_dataframe[INL][topic_dataframe['rank'] <= 50].mean(skipna=True)

        row_data = [{'topic': topic_dataframe['topic'][0], 'INL': INL, 'Top 1': top_1, 'Top 3': top_3, 'Top 5': top_5,
                     'Top 10': top_10,'Top 20': top_20, 'Top 50': top_50, 'Input': input_bias}]
        df_temp = pd.DataFrame(row_data)
        df_temp = df_temp.reindex(columns=['topic', 'INL', 'Top 1', 'Top 3', 'Top 5', 'Top 10', 'Top 20', 'Top 50',
                                           'Input'])
        df = df.append(df_temp, ignore_index=True)
    return df


def plot_averages(df, feature):
    """
    The feature is one of the INL features.  This function plots charts based on a select feature, such as readability.

    :param df: The dataframe (results) contining the top N results for different search results
    :param feature: An Information Nutrition Label
    :return:
    """

    data_by_feature = results[:][results['INL'] == feature]
    print(data_by_feature)

    ax = data_by_feature.plot(x='topic', y=['Top 1', 'Input'], kind='bar')
    plt.xticks(rotation=90)
    plt.title(f'{feature}'+' for top search result and input bias for each query')
    plt.ylabel('Ease of Reading')
    plt.xlabel('Query')
    # ax.set_ylim(1, 150)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), shadow=True, ncol=2)
    plt.show()

    # plt.barh(y='topic', width=['Top 1','Input'], data=data_by_feature)
    # plt.title(f'{feature}'+' for top search result and input bias for each query')
    # plt.xlabel('Ease of Reading')
    # plt.ylabel('Query')
    # plt.show()

    data_by_feature.index = data_by_feature['topic']
    data_by_feature.plot.barh(y=['Top 1', 'Input'])
    plt.title(f'{feature}'+' search results by query')
    plt.xlabel('Ease of Reading')
    plt.ylabel('Query')
    plt.legend(loc='upper center', bbox_to_anchor=(1.0, 0.5), shadow=True, ncol=1)
    plt.show()


if __name__ == '__main__':
    data = pd.read_csv(os.path.join(config.get_app_root(), "evaluation", 'politics_topics.csv'))
    results = pd.DataFrame({}, columns=['topic', 'INL', 'Top 1', 'Top 3', 'Top 5', 'Top 10', 'Top 20', 'Top 50',
                                        'Input'])
    features = ['readability', 'sentence_level_sentiment', 'sentence_level_objectivity', 'bias', 'credibility',
                'trust_metric', 'google_page_rank', 'alexa_reach_rank']
    # for topic in ['ClimateChange']:
    #     topic_data = data[data['topic'] == topic]
    #     create_scatter_plots(topic_data, features)
    #     feature_comparison(topic_data, features)
    for topic in data.topic.unique():
        topic_data = data[data['topic'] == topic]
        # create_scatter_plots(topic_data, features)
        results = get_output_averages(topic_data, features, results)

    # todo insert spaces in topic titles

    plot_averages(results, 'readability')

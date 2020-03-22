import pandas as pd
import os
import config
import matplotlib.pyplot as plt
import numpy as np
import re


def create_scatter_plots(topic_data, features):
    # todo: Add regression line
    # todo: Add title for topic_data
    for feature in features:
        ax = topic_data.plot(x=feature, y='rank', kind='scatter')
        ax.set_xlim(1, 101)
        ax.set_ylim(1, 101)
        plt.xticks(rotation=90)
        plt.show()


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


def plot_top1_input(df, feature):
    """
    This function plots INL score of top 1 search result compared to average of entire set of search results based on an
    INL feature, such as readability.

    :param df: The dataframe (results) contining the top N results for different search results
    :param feature: An Information Nutrition Label
    :return:
    """

    data_by_feature = df[:][df['INL'] == feature]

    data_by_feature.index = data_by_feature['topic']
    data_by_feature.plot.barh(y=['Top 1', 'Input'])
    # plt.title(f'{feature}'+' search results by query')
    plt.title(f'Ease of Reading search results by query')
    plt.xlabel('Ease of Reading')
    plt.ylabel('Query')
    plt.legend(loc='upper center', bbox_to_anchor=(1.0, 0.5), shadow=True, ncol=1)
    plt.show()


def plot_topN_averages(df, feature):
    """
    The feature is one of the INL features.  This function plots INL score of top N search results for a feature.
    :param df:
    :param feature:
    :return:
    """
    data_by_feature = df[:][df['INL'] == feature]

    data = [{'INL': data_by_feature['INL'][0],
             'Top 1': data_by_feature['Top 1'].mean(skipna=True),
             'Top 3': data_by_feature['Top 3'].mean(skipna=True),
             'Top 5': data_by_feature['Top 5'].mean(skipna=True),
             'Top 10': data_by_feature['Top 10'].mean(skipna=True),
             'Top 20': data_by_feature['Top 20'].mean(skipna=True),
             'Top 50': data_by_feature['Top 50'].mean(skipna=True),
             'Input': data_by_feature['Input'].mean(skipna=True)}]
    avg_df = pd.DataFrame(data)
    print(avg_df)

    # todo plots
    # ax = data_by_feature.plot(x='topic', y=['Top 1', 'Input'], kind='bar')
    # plt.xticks(rotation=90)
    # plt.title(f'{feature}'+' for top search result and input bias for each query')
    # plt.ylabel('Ease of Reading')
    # plt.xlabel('Query')
    # # ax.set_ylim(1, 150)
    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), shadow=True, ncol=2)
    # plt.show()


def reformat_topics(str):
    """
    Insert spaces into topic titles
    :param str:
    :return:
    """
    return re.sub(r"(\w)([A-Z])", r"\1 \2", str)


if __name__ == '__main__':
    data_poli = pd.read_csv(os.path.join(config.get_app_root(), "evaluation", 'politics_topics.csv'))
    data_econ = pd.read_csv(os.path.join(config.get_app_root(), "evaluation", 'economics_topics.csv'))

    results_econ = pd.DataFrame({}, columns=['topic', 'INL', 'Top 1', 'Top 3', 'Top 5', 'Top 10', 'Top 20', 'Top 50',
                                             'Input'])
    results_poli = pd.DataFrame({}, columns=['topic', 'INL', 'Top 1', 'Top 3', 'Top 5', 'Top 10', 'Top 20', 'Top 50',
                                             'Input'])
    features = ['readability', 'sentence_level_sentiment', 'sentence_level_objectivity', 'bias', 'credibility',
                'trust_metric', 'google_page_rank', 'alexa_reach_rank']
    data_list = [data_poli, data_econ]

    # Create Dataframe containing search result averages
    for data in data_list:
        for topic in data.topic.unique():
            topic_data = data[data['topic'] == topic]
            # create_scatter_plots(topic_data, features)
            if data is data_poli:
                results_poli = get_output_averages(topic_data, features, results_poli)
            else:
                results_econ = get_output_averages(topic_data, features, results_econ)

    results_poli['topic'] = results_poli['topic'].apply(reformat_topics)
    results_econ['topic'] = results_econ['topic'].apply(reformat_topics)
    plot_top1_input(results_poli, 'readability')
    plot_top1_input(results_econ, 'readability')

    results = results_econ.append(results_poli, ignore_index=True)
    plot_topN_averages(results, 'readability')

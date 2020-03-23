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


def get_output_bias(df, INL):
    """
    Calculate output bias for top 3, 5, 10, 20 , 50 results per topic and return dataframe
    :param df: data of all INL scores
    :param feature: one INL feature
    :return: dataframe of output_bias_df
    """
    output_bias_df = pd.DataFrame()
    for topic in df.topic.unique():
        topic_df = df[df['topic'] == topic]
        output_bias_dict = {'topic': topic,'INL': INL}
        for j in [3, 5, 10, 20, 50]:
            output_bias = 0
            i = 1
            while i <= j:
                bias = topic_df['readability'][topic_df['rank'] <= i].mean(skipna=True)
                if bias is not np.nan:
                    output_bias = output_bias + bias
                i = i + 1
            output_bias_dict['OB Rank '+str(j)] = output_bias/j

        output_bias_dict['Input Bias'] = topic_df['readability'].mean(skipna=True)

        # Append row to output bias dataframe
        df_temp = pd.DataFrame([output_bias_dict])
        output_bias_df = output_bias_df.append(df_temp, ignore_index=True)
    return output_bias_df


def get_output_averages(topic_dataframe, feature, df):
    """
    A small sub-table (sub-dataframe) of all the INL scores for one topic is passed into the function (topic_dataframe).
    Then, for each topic, the averages of the top N search results are calculated and one row (per topic) is appended to
    a new dataframe (df).

    :param topic_dataframe: contains INL scores from just one topic
    :param feature: a list of INL features
    dataframe.
    :param df: is either an empty dataframe or a dataframe containing the results from a previous call to this function.
    :return: df
    """
    topic_dataframe.reset_index(drop=True, inplace=True)
    for INL in feature:
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
    data_by_feature.plot.barh(y=['Top 1', 'Top 10', 'Input'])
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

    ax = avg_df.plot(x='INL', y=['Top 1', 'Top 3', 'Top 5', 'Top 10', 'Top 20', 'Top 50', 'Input'], kind='bar')
    plt.xticks([])
    plt.title(f'Average search results')
    plt.ylabel('Ease of Reading')
    plt.xlabel('Top N Results')
    plt.show()


def plot_output_bias(data_by_feature):
    """

    :param df:
    :return:
    """
    # ax = df.plot(x='INL', y=['OB Rank 3', 'OB Rank 5', 'OB Rank 10', 'OB Rank 20', 'OB Rank 50', 'OB Rank 50', 'Input Bias'], kind='bar')
    # plt.xticks([])
    # plt.title(f'Average search results')
    # plt.ylabel('Ease of Reading')
    # plt.xlabel('Top N Results')
    # plt.show()

    data_by_feature.index = data_by_feature['topic']
    data_by_feature.plot.barh(y=['OB Rank 10', 'OB Rank 20', 'Input Bias'])
    # plt.title(f'{feature}'+' search results by query')
    plt.title(f'Ease of Reading search results by query')
    plt.xlabel('Ease of Reading')
    plt.ylabel('Query')
    plt.legend(loc='upper center', bbox_to_anchor=(1.0, 0.5), shadow=True, ncol=1)
    plt.show()


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

    # Create dataframe containing output bias per topic
    results = data_poli.append(data_econ, ignore_index=True)

    # for one feature...
    output_bias_df = get_output_bias(results, 'readability')
    plot_output_bias(output_bias_df)

    # Create Dataframe containing search result averages
    for data in data_list:
        for topic in data.topic.unique():
            topic_data = data[data['topic'] == topic]
            # create_scatter_plots(topic_data, features)
            # create_scatter_plots(topic_data, ['readability'])
            if data is data_poli:
                results_poli = get_output_averages(topic_data, features, results_poli)
            else:
                results_econ = get_output_averages(topic_data, features, results_econ)

    results_poli['topic'] = results_poli['topic'].apply(reformat_topics)
    results_econ['topic'] = results_econ['topic'].apply(reformat_topics)
    # plot_top1_input(results_poli, 'readability')
    # plot_top1_input(results_econ, 'readability')
    #
    # average_results = results_econ.append(results_poli, ignore_index=True)
    # plot_topN_averages(average_results, 'readability')


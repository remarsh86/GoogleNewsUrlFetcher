import pandas as pd
import os
import config
import matplotlib.pyplot as plt
import numpy as np
import re
from scipy.stats import ks_2samp

FEATURE = 'ease_of_reading'
# TODO: replace string in document with FEATURE


def get_output_bias(topic_df, features, topic):
    """
    Calculate output bias for top 3, 5, 10, 20 , 50 results per topic and return dataframe

    When calculated output bias, it's important to consider that some ranked documents could not be evaluated by newscan.
    Therefore, for output bias rank 3, there may only be two documents in this category.  The output bias calculation
    had to be adjusted to account for this.  In the loop, there may be less than j bias scores so divide by count
    (which counts bias score) instead.  Also, bias scores (the mean of the documents to rank i) should not be calculated
    for an i that is not in the ranking.  Example: ranking = [1, 3].  I.e. Document with rank 2 could not be evaluated by
    NewsScan.  So the output bias is (b_1 + mean(b_1, b_2))/2, where b_i is the bias score of the document at rank i.
    :param df: data of all INL scores
    :param feature: one INL feature
    :return: dataframe of output_bias_df
    """

    output_bias_df = pd.DataFrame()
    for INL in features:
        output_bias_dict = {'topic': topic,'INL': INL}
        for j in [3, 5, 10, 20, 50]:
            output_bias = 0
            i = 1
            count = 0
            while i <= j:
                if i in topic_df['rank'][topic_df['rank'] <= j].values:
                    bias = topic_df[INL][topic_df['rank'] <= i].mean(skipna=True)
                    if bias is not np.nan:
                        output_bias = output_bias + bias
                        count = count + 1
                i = i + 1
            if count != 0:
                # There may be less than j bias scores so divide by count (which counts bias score) instead.
                output_bias_dict['OB Rank ' + str(j)] = output_bias / count
            else:
                output_bias_dict['OB Rank ' + str(j)] = np.nan

        output_bias_dict['Input Bias'] = topic_df[INL].mean(skipna=True)

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
        # Choose item with rank = 1 for top 1. If doesn't exist, then choose first item available in ranking as top 1
        if topic_dataframe[INL][topic_dataframe['rank'] == 1] is not None:
            top_1 = topic_dataframe[INL][0]
        # print(f'Input bias of {INL} in {topic}: {topic_dataframe[INL].mean(skipna=True)}')
        input_bias = topic_dataframe[INL].mean(skipna=True)
        top_3 = topic_dataframe[INL][topic_dataframe['rank'] <= 3].mean(skipna=True)
        top_5 = topic_dataframe[INL][topic_dataframe['rank'] <= 5].mean(skipna=True)
        top_10 = topic_dataframe[INL][topic_dataframe['rank'] <= 10].mean(skipna=True)
        top_20 = topic_dataframe[INL][topic_dataframe['rank'] <= 20].mean(skipna=True)
        top_50 = topic_dataframe[INL][topic_dataframe['rank'] <= 50].mean(skipna=True)

        row_data = [{'topic': topic_dataframe['topic'][0], 'INL': INL, 'Top 1': top_1, 'Top 3': top_3, 'Top 5': top_5,
                     'Top 10': top_10,'Top 20': top_20, 'Top 50': top_50, 'Input Bias': input_bias}]
        df_temp = pd.DataFrame(row_data)
        df_temp = df_temp.reindex(columns=['topic', 'INL', 'Top 1', 'Top 3', 'Top 5', 'Top 10', 'Top 20', 'Top 50',
                                           'Input Bias'])
        df = df.append(df_temp, ignore_index=True)
    return df


def get_averages_rank_N(data_list, features):
    # For analysis of averages in top N results
    results_econ = pd.DataFrame({}, columns=['topic', 'INL', 'Top 1', 'Top 3', 'Top 5', 'Top 10', 'Top 20', 'Top 50',
                                             'Input'])
    results_poli = pd.DataFrame({}, columns=['topic', 'INL', 'Top 1', 'Top 3', 'Top 5', 'Top 10', 'Top 20', 'Top 50',
                                             'Input'])
    # For analysis of output bias
    output_bias_econ = pd.DataFrame()
    output_bias_poli = pd.DataFrame()

    # Create Dataframe containing search result averages
    for data in data_list:
        for topic in data.topic.unique():
            topic_data = data[data['topic'] == topic]
            if data is data_list[0]:
                output_bias_poli = output_bias_poli.append(get_output_bias(topic_data, features, topic))
                results_poli = get_output_averages(topic_data, features, results_poli)
            else:
                output_bias_econ = output_bias_econ.append(get_output_bias(topic_data, features, topic))
                results_econ = get_output_averages(topic_data, features, results_econ)

    results_poli['topic'] = results_poli['topic'].apply(reformat_topics)
    results_econ['topic'] = results_econ['topic'].apply(reformat_topics)
    output_bias_poli['topic'] = output_bias_poli['topic'].apply(reformat_topics)
    output_bias_econ['topic'] = output_bias_econ['topic'].apply(reformat_topics)
    return results_poli, results_econ, output_bias_poli, output_bias_econ


def remove_queries(res):
    """
    Remove queries from dataset if NewsScan could not calculate INL scores for the top ranked search
    result.
    :param res: dataframe of NewsScan search results to n queries
    :return:
    """
    top_1 = res[:][res['rank'] == 1]
    data = res[:][res['topic'].isin(top_1['topic'].values)]
    return data


def test_distribution_similarity(feature, g1, g2):
    series1 = g1.groupby(feature).count()['topic']
    series2 = g2.groupby(feature).count()['topic']
    print(ks_2samp(series1, series2))


def get_feature_distribution(feature, res, title):
    """
    Plot the distribution of INL scores
    :param feature: an INL like 'ease_of_reading'
    :param res: Dataframe of results
    :return:
    """
    label = feature.replace('_', ' ').title().replace('Of', 'of')
    df = res.groupby(feature).count()[['topic']]
    print(df)
    df.rename(columns={'topic': 'Search Results'}, inplace=True)
    df.rename(index={20.0: str(20.0)+' (C2)', 40.0: str(40.0)+' (C1)', 60.0: str(60.0)+' (B2)', 80.0: str(80.0)+' (B1)',
                     100.0: str(100.0)+' (A2)' }, inplace=True)
    df.plot(kind='bar')
    plt.ylabel('Frequency of Occurrences')
    plt.xlabel(label)
    plt.title(f'{label} Distribution for {title}')
    plt.show()


def create_scatter_plots(topic_data, features):
    """
    Plots results for the search results (from all 100 results) for each feature for one query.
    :param topic_data: a query string; ex: "RefugeeCrisis"
    :param features:  ['readability', 'sentence_level_sentiment', 'sentence_level_objectivity', 'bias', 'credibility',
                'trust_metric', 'google_page_rank', 'alexa_reach_rank']
    :return: None
    """
    # CURRENTLY NOT BEING USED
    # todo: Add regression line
    # todo: Add title for topic_data
    for feature in features:
        ax = topic_data.plot(x=feature, y='rank', kind='scatter')
        ax.set_xlim(1, 101)
        ax.set_ylim(1, 101)
        plt.xticks(rotation=90)
        plt.show()


def create_feature_scatterplot(feature, results):
    """
    Plots results for the search results of one INL Label for all queries.

    :param feature:
    :param results:
    :return: None
    """
    label = reformat_topics(feature.replace('_', ' '))
    x = results['rank']
    y = results[feature]
    plt.scatter(x, y, s=70, alpha=0.03)
    plt.ylim((1, 101))
    plt.xlim((1, 101))
    plt.title(f'{label} Results for all Queries')
    plt.ylabel(label)
    plt.xlabel('Rank')
    plt.show()

    # results10 = results[:][results['rank'] <= 10]
    # x = results10['rank']
    # y = results10[feature]
    # plt.scatter(x, y, s=70, alpha=0.03)
    # plt.ylim((1, 101))
    # plt.xlim((1, 10))
    # plt.show()


def create_feature_density_plot(feature, results):
    """
    Create a scatter plot of a feature (INL) for each query
    :param feature: (INL) feature
    :param results: dataframe of results
    :return:
    """
    # CURRENTLY NOT BEING USED
    for topic in results.topic.unique():
        topic_subset = results[results['topic'] == topic]
        ax = topic_subset.plot(y=feature, x='rank', kind='line')
        ax.set_xlim(1, 101)
        ax.set_ylim(1, 101)
        plt.title(f'Ease of Reading Results for {topic}')
        plt.ylabel('Ease of Reading')
        plt.xlabel('Rank')
        plt.show()


def plot_top1_input(df, feature, title):
    """
    This function plots INL score of top 1 search result compared to average of entire set of search results based on an
    INL feature, such as readability.

    :param df: The dataframe (results) contining the top N results for different search results
    :param feature: An Information Nutrition Label
    :return:
    """

    data_by_feature = df[:][df['INL'] == feature]

    data_by_feature.index = data_by_feature['topic']
    data_by_feature.plot.barh(y=['Top 1','Input Bias'])
    plt.title(f'Ease of Reading Search Results {title}')
    plt.xlabel('Ease of Reading')
    plt.ylabel('Query')
    plt.xticks([0, 20, 40, 60, 80, 100])
    plt.legend(loc='upper center', bbox_to_anchor=(1.0, 1.0), shadow=True, ncol=1)
    plt.show()


def plot_topN_averages(df, feature, title):
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
             'Input Bias': data_by_feature['Input Bias'].mean(skipna=True)}]
    avg_df = pd.DataFrame(data)

    ax = avg_df.plot(x='INL', y=['Top 1', 'Top 3', 'Top 5', 'Top 10', 'Top 20', 'Top 50', 'Input Bias'], kind='bar')
    plt.xticks([])
    plt.title(f'Average search results {title}')
    plt.ylabel('Ease of Reading')
    plt.xlabel('Top N Results')
    plt.show()


def plot_output_bias_by_query(df, feature, title):
    """

    :param df:
    :return:
    """
    data_by_feature = df[:][df['INL'] == feature]

    data_by_feature.index = data_by_feature['topic']
    data_by_feature.plot.barh(y=['OB Rank 10', 'OB Rank 20', 'Input Bias'])
    plt.title(f'Ease of Reading output bias for rank N for {title}')
    plt.xlabel('Ease of Reading')
    plt.ylabel('Query')
    plt.legend(loc='upper center', bbox_to_anchor=(1.0, 0.5), shadow=True, ncol=1)
    plt.show()


def plot_average_output_bias(df, feature, title):
    """

    :param df:
    :return:
    """
    data_by_feature = df[:][df['INL'] == feature]
    data_by_feature.reset_index(drop=True, inplace=True)

    data = [{'INL': data_by_feature['INL'][0],
             'OB Rank 3': data_by_feature['OB Rank 3'].mean(skipna=True),
             'OB Rank 5': data_by_feature['OB Rank 5'].mean(skipna=True),
             'OB Rank 10': data_by_feature['OB Rank 10'].mean(skipna=True),
             'OB Rank 20': data_by_feature['OB Rank 20'].mean(skipna=True),
             'OB Rank 50': data_by_feature['OB Rank 50'].mean(skipna=True),
             'Input Bias': data_by_feature['Input Bias'].mean(skipna=True)}]
    avg_df = pd.DataFrame(data)

    ax = avg_df.plot(x='INL', y=['OB Rank 3', 'OB Rank 5', 'OB Rank 10', 'OB Rank 20', 'OB Rank 50', 'Input Bias'], kind='bar')
    plt.xticks([])
    plt.title(f'Average Output Bias Results {title}')
    plt.ylabel('Ease of Reading')
    plt.xlabel('Output Bias for Rank N')
    plt.show()

    plot_ranking_bias(avg_df, title)


def plot_ranking_bias(df, title):
    """
    Calculate ranking bias: RB(q, r) =OB(q, r)−IB(q) for each rank (3, 5, 10, 20, 50).
    :param df:
    :param feature:
    :return:
    """
    df['INL'][0] = df['INL'][0].capitalize()
    df['Ranking Bias R3'] = df['OB Rank 3'] - df['Input Bias']
    df['Ranking Bias R5'] = df['OB Rank 5'] - df['Input Bias']
    df['Ranking Bias R10'] = df['OB Rank 10'] - df['Input Bias']
    df['Ranking Bias R20'] = df['OB Rank 20'] - df['Input Bias']
    df['Ranking Bias R50'] = df['OB Rank 50'] - df['Input Bias']
    ax = df.plot(x='INL', y=['Ranking Bias R3', 'Ranking Bias R5', 'Ranking Bias R10', 'Ranking Bias R20', 'Ranking Bias R50'],
                     kind='bar')
    feature = df['INL'][0]
    plt.xticks([])
    # plt.title(f'{feature} Ranking Bias Results {title}')
    # plt.ylabel(f'{feature} Ranking Bias')
    plt.title(f'Ease of Reading Ranking Bias Results {title}')
    plt.ylabel(f'Ease of Reading Ranking Bias')
    plt.xlabel('Ranking Bias for Rank N')
    plt.show()


def reformat_topics(str):
    """
    Insert spaces into topic titles
    :param str:
    :return:
    """
    return re.sub(r"(\w)([A-Z])", r"\1 \2", str)


if __name__ == '__main__':
    data_poli = pd.read_csv(os.path.join(config.get_app_root(), "evaluation", 'politics_topics_2.csv'))
    data_econ = pd.read_csv(os.path.join(config.get_app_root(), "evaluation", 'economics_topics_2.csv'))

    # Remove queries without document at ranking 1
    data_poli = remove_queries(data_poli)
    data_econ = remove_queries(data_econ)

    data_list = [data_poli, data_econ]
    features = ['ease_of_reading', 'sentence_level_sentiment', 'sentence_level_objectivity', 'bias', 'credibility',
                'trust_metric', 'google_page_rank', 'alexa_reach_rank']

    results_poli, results_econ, output_bias_poli, output_bias_econ = get_averages_rank_N(data_list, features)

    # # Analyze output bias
    plot_output_bias_by_query(output_bias_poli, FEATURE, "for Political Queries")
    plot_output_bias_by_query(output_bias_econ, FEATURE, "for Economic Queries")
    output_bias = output_bias_econ.append(output_bias_poli, ignore_index=True)
    plot_average_output_bias(output_bias, FEATURE, "for all Queries")

    # Analyze average results
    plot_top1_input(results_poli, FEATURE, "for Political Queries")
    plot_top1_input(results_econ, FEATURE, "for Economic Queries")
    average_results = results_econ.append(results_poli, ignore_index=True)
    plot_topN_averages(average_results, FEATURE, 'for all Queries')

    # Create Scatterplot for each feature containing all results
    results = data_poli.append(data_econ, ignore_index=True)
    for feature in features:
        create_feature_scatterplot(feature, results)


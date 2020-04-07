import pandas as pd
import os
import config
import evaluation.analysis as analysis


def split_on_readability(res):
    """
    Split the results from NewsScan into two groups: group A contains all the data where the first search result has a
    Readability score less than or equal to 60.  group B contains the rest of the data.
    top_1 = subset of data containing top ranked results only.
    :param res:
    :return: return two dataframes which split the data
    """
    top_1= res[:][res['rank'] == 1]
    feature_average_df = res.groupby('topic').mean()[['ease_of_reading']]
    group_A = top_1[:][(top_1['ease_of_reading'] <= 60)]
    group_B = top_1[:][top_1['ease_of_reading'] > 60]
    topics_A = group_A['topic'].values
    topics_B = group_B['topic'].values
    clean_topics_B = []
    # Eliminate queries from group_B where input bias is more than top 1 search result
    for topic in topics_B:
        # Get average score for topic
        input_bias = feature_average_df[[x.startswith(topic) for x in feature_average_df.index]]['ease_of_reading'][0]
        top_score = top_1[:][top_1['topic']==topic]['ease_of_reading'].iloc[0]
        if input_bias <= top_score:
            clean_topics_B.append(topic)
    return res[res['topic'].isin(topics_A)], res[res['topic'].isin(clean_topics_B)]


if __name__ == '__main__':
    data_poli = pd.read_csv(os.path.join(config.get_app_root(), "evaluation", 'politics_topics_2.csv'))
    data_econ = pd.read_csv(os.path.join(config.get_app_root(), "evaluation", 'economics_topics_2.csv'))

    features = ['ease_of_reading', 'sentence_level_sentiment', 'sentence_level_objectivity', 'bias', 'credibility',
                'trust_metric', 'google_page_rank', 'alexa_reach_rank']

    # analyze data split on top results for readability results
    results = data_poli.append(data_econ, ignore_index=True)

    # analyze data split on top results for readability results
    group_A, group_B = split_on_readability(results)

    # Get scatter Plot
    analysis.create_feature_scatterplot('ease_of_reading', group_A)
    analysis.create_feature_scatterplot('ease_of_reading', group_B)

    # Get distribution of ease of reading groups
    analysis.get_feature_distribution('ease_of_reading', group_A, "Group A")
    analysis.get_feature_distribution('ease_of_reading', group_B, "Group B")
    analysis.test_distribution_similarity('ease_of_reading', group_A, group_B)

    split_data_list = [group_A, group_B]
    results_A, results_B, output_bias_A, output_bias_B = analysis.get_averages_rank_N(split_data_list, features)

    # # Analyze output bias
    # analysis.plot_output_bias_by_query(output_bias_A, 'ease_of_reading', 'for Group A')
    # analysis.plot_average_output_bias(output_bias_A, 'ease_of_reading', ' for Group A')
    # analysis.plot_output_bias_by_query(output_bias_B, 'ease_of_reading', 'for Group B')
    # analysis.plot_average_output_bias(output_bias_B, 'ease_of_reading', ' for Group B')

    # # Analyze average results
    analysis.plot_top1_input(results_A, 'ease_of_reading', 'for Group A')
    analysis.plot_topN_averages(results_A, 'ease_of_reading', 'for Group A')
    analysis.plot_top1_input(results_B, 'ease_of_reading', 'for Group B')
    analysis.plot_topN_averages(results_B, 'ease_of_reading', 'for Group B')



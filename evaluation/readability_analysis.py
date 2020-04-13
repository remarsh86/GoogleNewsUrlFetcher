import pandas as pd
import os
import config
import evaluation.analysis as analysis
import matplotlib.pyplot as plt

# FEATURE = 'content_length'
FEATURE = 'ease_of_reading'


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


def get_feature_bargraph(feature, res, title):
    """
    Plot the distribution of INL scores. The feature should be 'ease_of_reading'...(?)
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


def get_histogram(res, feature, title):
    """
    ****Used for analysis of content length?
    Bin data into discrete groups to visualize as bar graph
    :param: feature: INL feature
    :return:
    """
    # reduce to only results with a character length of 50000
    reduced_res = res[:][res[feature] <= 50000]
    ax = reduced_res.hist(column=feature, bins=50, grid=False)

    label = feature.replace('_', ' ').title().replace('Of', 'of')
    # plt.yticks([0, 50, 100, 150, 200, 250, 300, 350, 400])
    plt.title(f'{label} Results for {title}')
    plt.ylabel(f'Number of Occurrences')
    plt.xlabel(f'{label}')
    plt.show()

def analyze_document_length(res, group_A, group_B):
    c = res['ease_of_reading'].corr(res['content_length'])
    print("Does readability correlate with content length? Pearson's Method: ", c )


if __name__ == '__main__':
    data_poli = pd.read_csv(os.path.join(config.get_app_root(), "evaluation", 'politics_topics_2.csv'))
    data_econ = pd.read_csv(os.path.join(config.get_app_root(), "evaluation", 'economics_topics_2.csv'))

    features = ['ease_of_reading', 'sentence_level_sentiment', 'sentence_level_objectivity', 'bias', 'credibility',
                'trust_metric', 'google_page_rank', 'alexa_reach_rank', 'content_length']

    # analyze data split on top results for readability results
    results = data_poli.append(data_econ, ignore_index=True)

    # analyze data split on top results for readability results
    group_A, group_B = split_on_readability(results)

    # Analyze Readability in terms of content length
    analyze_document_length(results, group_A, group_B)

    # Get scatter Plot
    analysis.create_feature_scatterplot('ease_of_reading', group_A, 'for Group A')
    analysis.create_feature_scatterplot('ease_of_reading', group_B, 'for Group B')

    # # Analyze document length
    # get_histogram(group_A, FEATURE, 'for Group A')
    # get_histogram(group_B, FEATURE, 'for Group B')

    # Get distribution of ease of reading groups
    # get_feature_bargraph('ease_of_reading', group_A, "Group A")
    # get_feature_bargraph('ease_of_reading', group_B, "Group B")
    # analysis.test_distribution_similarity('ease_of_reading', group_A, group_B)

    # # Create scatterplot of INL results
    # analysis.create_feature_scatterplot(FEATURE, group_A, "Group A")
    # analysis.create_feature_scatterplot(FEATURE, group_B, "Group B")

    split_data_list = [group_A, group_B]
    results_A, results_B, output_bias_A, output_bias_B = analysis.get_averages_rank_N(split_data_list, features)

    # # Analyze output bias
    # analysis.plot_output_bias_by_query(output_bias_A, 'ease_of_reading', 'for Group A')
    # analysis.plot_average_output_bias(output_bias_A, 'ease_of_reading', ' for Group A')
    # analysis.plot_output_bias_by_query(output_bias_B, 'ease_of_reading', 'for Group B')
    # analysis.plot_average_output_bias(output_bias_B, 'ease_of_reading', ' for Group B')

    # # Analyze average results
    analysis.plot_top1_input(results_A, FEATURE, 'for Group A')
    analysis.plot_topN_averages(results_A, FEATURE, 'for Group A')
    analysis.plot_top1_input(results_B, FEATURE, 'for Group B')
    analysis.plot_topN_averages(results_B, FEATURE, 'for Group B')

    analysis.plot_top1_input(results_A, 'content_length', 'for Group A')
    analysis.plot_topN_averages(results_A, 'content_length', 'for Group A')
    analysis.plot_top1_input(results_B, 'content_length', 'for Group B')
    analysis.plot_topN_averages(results_B, 'content_length', 'for Group B')



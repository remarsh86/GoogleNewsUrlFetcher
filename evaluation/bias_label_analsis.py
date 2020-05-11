import pandas as pd
import os
import config
import evaluation.analysis as analysis

FEATURE = 'bias_label'


def get_topic_results(query, data_df):
    return data_df[data_df['topic'] == query]


def compare_domains(results):
    refugee_crisis_results = get_topic_results('RefugeeCrisis', results)
    refugee_domains = refugee_crisis_results['domain'].unique()
    medicare_results = get_topic_results('MedicareForAll', results)
    medicare_domains = medicare_results['domain'].unique()
    unique_refugee_domains = []
    unique_medicare_domains = []
    same_domains = []
    for domain in refugee_domains:
        if domain not in medicare_domains:
            unique_refugee_domains.append(domain)
    for domain in medicare_domains:
        if domain not in refugee_domains:
            unique_medicare_domains.append(domain)
    for domain in medicare_domains:
        if domain in refugee_domains:
            same_domains.append(domain)
    print("Unique domains to Refugee: ", unique_refugee_domains)
    print("Unique domains to Medicare: ", unique_medicare_domains)
    print("Shared domains to both: ", same_domains)


def compare_domains_bias_labels(results, label1, label2):
    """
    Compare two labels such as Left and Far Right for the given feature FEATURE
    :param results: dataframe of results
    :param label1: a group label
    :param label2: a group label
    :return: None
    """
    label1_data = results[results[FEATURE] == label1]
    label2_data = results[results[FEATURE] == label2]

    label1_domains = label1_data['domain'].unique()
    label2_domains = label2_data['domain'].unique()
    print(len(label1_data['domain']), "Unique: ", len(label1_domains))
    print(len(label2_data['domain']), "Unique: ", len(label2_domains))

    unique_label1_domains = list(set(label1_domains) - set(label2_domains))
    unique_label2_domains = list(set(label2_domains) - set(label1_domains))
    print("Left domains ", len(unique_label1_domains), unique_label1_domains)
    print("Far Right domains ", len(unique_label2_domains), unique_label2_domains)


if __name__ == '__main__':
    data_poli = pd.read_csv(os.path.join(config.get_app_root(), "evaluation", 'politics_topics_2.csv'))
    data_econ = pd.read_csv(os.path.join(config.get_app_root(), "evaluation", 'economics_topics_2.csv'))

    # Convert political bias labels to numeric labels
    data_poli = analysis.change_poli_labels(FEATURE, data_poli)
    data_econ = analysis.change_poli_labels(FEATURE, data_econ)

    # Remove queries without document at ranking 1
    data_poli = analysis.remove_queries(data_poli)
    data_econ = analysis.remove_queries(data_econ)

    # The following INL features are numeric in value and can be counted, aggregated, plotted, etc..
    features = ['ease_of_reading', 'sentence_level_sentiment', 'sentence_level_objectivity', 'bias', 'credibility',
                'trust_metric', 'google_page_rank', 'alexa_reach_rank', 'content_length', 'bias_label']

    # Create Scatterplot for each feature containing all results
    results = data_poli.append(data_econ, ignore_index=True)
    print("Number of unique domains and article count: ", results['domain'].value_counts())

    # Begin analysis of individual queries
    print('Refugee Crisis')
    refugee_crisis_results = get_topic_results('RefugeeCrisis', results)
    # Get statistical parity and disparate impact stacked bar charts
    stat_parity_df_1 = analysis.get_distribution_df(FEATURE, refugee_crisis_results, 'for Topic Refugee Crisis')
    disparate_impact_df_1 = analysis.get_dist_percent_df(FEATURE, stat_parity_df_1)
    analysis.print_to_excel(stat_parity_df_1, 'RefugeeCrisis')
    analysis.plot_statistical_parity(FEATURE, stat_parity_df_1, 'for Topic Refugee Crisis')
    analysis.plot_disparate_impact(FEATURE, disparate_impact_df_1, ' (Topic Refugee Crisis)')

    print()
    print('Medicare For All: ')
    medicare_results = get_topic_results('MedicareForAll', results)
    # Get statistical parity and disparate impact stacked bar charts
    stat_parity_df = analysis.get_distribution_df(FEATURE, medicare_results, 'for Topic Medicare for All')
    disparate_impact_df = analysis.get_dist_percent_df(FEATURE, stat_parity_df)
    analysis.print_to_excel(stat_parity_df, 'MedicareForAll')
    analysis.plot_statistical_parity(FEATURE, stat_parity_df, 'for Topic Medicare for All')
    analysis.plot_disparate_impact(FEATURE, disparate_impact_df, ' (Topic Medicare for All)')

    # Study domains with Left and Far Right label
    compare_domains_bias_labels(results, -1, 2)

    # Examine difference in political and economic queries
    data_list = [data_poli, data_econ]
    results_poli, results_econ, output_bias_poli, output_bias_econ, label_count_poli, label_count_econ = \
        analysis.get_averages_rank_N(data_list, features)
    analysis.plot_topN_averages(results_poli, FEATURE, 'for Political Queries')
    analysis.plot_topN_averages(results_econ, FEATURE, 'for Economic Queries')

    # Analyze output bias
    analysis.plot_average_output_bias(output_bias_poli, FEATURE, 'for Political Queries')
    analysis.plot_average_output_bias(output_bias_econ, FEATURE, 'for Economic Queries')

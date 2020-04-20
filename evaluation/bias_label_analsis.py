import pandas as pd
import os
import config
import evaluation.analysis as analysis
import matplotlib.pyplot as plt

FEATURE = 'bias_label'

def get_topic_results(query, data_df):
    # for topic in data_df.topic.unique():
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
    label1_data = results[results[FEATURE] == label1]
    label2_data = results[results[FEATURE] == label2]

    label1_domains = label1_data['domain'].unique()
    label2_domains = label2_data['domain'].unique()
    print(label1_domains)
    print(label2_domains)

    unique_label1_domains = list(set(label1_domains) - set(label2_domains))
    unique_label2_domains = list(set(label2_domains)-set(label2_domains))
    print(unique_label1_domains)
    print(unique_label2_domains)




if __name__ == '__main__':
    data_poli = pd.read_csv(os.path.join(config.get_app_root(), "evaluation", 'politics_topics_2.csv'))
    data_econ = pd.read_csv(os.path.join(config.get_app_root(), "evaluation", 'economics_topics_2.csv'))

    # Convert political bias labels to numeric labels
    data_poli = analysis.change_poli_labels(FEATURE, data_poli)
    data_econ = analysis.change_poli_labels(FEATURE, data_econ)

    # The following INL features are numeric in value and can be counted, aggregated, plotted, etc..
    features = ['ease_of_reading', 'sentence_level_sentiment', 'sentence_level_objectivity', 'bias', 'credibility',
                'trust_metric', 'google_page_rank', 'alexa_reach_rank', 'content_length', 'bias_label']

    # Create Scatterplot for each feature containing all results
    results = data_poli.append(data_econ, ignore_index=True)

    # Begin analysis of individual queries
    print('Refugee Crisis')
    refugee_crisis_results = get_topic_results('RefugeeCrisis', results)
    # Get statistical parity and disparate impact stacked bar charts
    stat_parity_df_1 = analysis.get_distribution_df(FEATURE, refugee_crisis_results, 'for Topic Refugee Crisis')
    disparate_impact_df_1 = analysis.get_dist_percent_df(FEATURE, stat_parity_df_1)
    analysis.print_to_excel(stat_parity_df_1, 'RefugeeCrisis')
    # analysis.print_to_excel(disparate_impact_df_1, 'RefugeeCrisis')
    analysis.plot_statistical_parity(FEATURE, stat_parity_df_1, 'for Topic Refugee Crisis')
    analysis.plot_disparate_impact(FEATURE, disparate_impact_df_1, ' (Topic Refugee Crisis)')

    # print()
    # print('Brexit')
    # brexit_results = get_topic_results('Brexit', results)
    # # Get statistical parity and disparate impact stacked bar charts
    # stat_parity_df_1 = analysis.get_distribution_df(FEATURE, brexit_results, 'for Topic Brexit')
    # disparate_impact_df_1 = analysis.get_dist_percent_df(FEATURE, stat_parity_df_1)
    # analysis.print_to_excel(stat_parity_df_1, 'Brexit')
    # analysis.plot_statistical_parity(FEATURE, stat_parity_df_1, 'for Topic Brexit')
    # analysis.plot_disparate_impact(FEATURE, disparate_impact_df_1, ' (Topic Brexit)')

    print()
    print('Medicare For All: ')
    medicare_results = get_topic_results('MedicareForAll', results)
    # Get statistical parity and disparate impact stacked bar charts
    stat_parity_df = analysis.get_distribution_df(FEATURE, medicare_results, 'for Topic Medicare for All')
    disparate_impact_df = analysis.get_dist_percent_df(FEATURE, stat_parity_df)
    analysis.print_to_excel(stat_parity_df, 'MedicareForAll')
    # analysis.print_to_excel(disparate_impact_df, 'MedicareForAll')
    analysis.plot_statistical_parity(FEATURE, stat_parity_df, 'for Topic Medicare for All')
    analysis.plot_disparate_impact(FEATURE, disparate_impact_df, ' (Topic Medicare for All)')

    # print()
    # print('Gun Debate: ')
    # gun_results = get_topic_results('GunDebate', results)
    # # Get statistical parity and disparate impact stacked bar charts
    # stat_parity_df = analysis.get_distribution_df(FEATURE, gun_results, 'for Topic Gun Debate')
    # disparate_impact_df = analysis.get_dist_percent_df(FEATURE, stat_parity_df)
    # analysis.print_to_excel(stat_parity_df, 'GunDebate')
    # # analysis.print_to_excel(disparate_impact_df, 'GunDebate')
    # analysis.plot_statistical_parity(FEATURE, stat_parity_df, 'for Topic Gun Debate')
    # analysis.plot_disparate_impact(FEATURE, disparate_impact_df, ' (Topic Gun Debate)')


    # Study domains related to Refugee Crisis and Medicare for All
    # compare_domains(results)
    compare_domains_bias_labels(results, 'Left', 'Far Right')

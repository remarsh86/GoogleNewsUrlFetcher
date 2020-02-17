"""
Purpose:
1. For each topic, feed each fetched url into NewsScan Software through the NewsScan domain and port:
http://ariadne.is.inf.uni-due.de:7999/nutrition?
2. Collect results (in json format) for each call to NewsScanner and save results to a csv file

The resulting csv file will contain 2000 rows, each row containing the resulting NewsScan scores for a url. The results
for each category will be saved in one csv file.


"""
import os
import config
import requests, json
import csv
import numpy as np
import logging

from exceptions.news_scan_error import NewsScanError

logging.basicConfig(filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')


def feed_news_scan():
    """The following opens the topics files containing the result set of URL's from Google News for each topic and feeds
    the URLS into the evaluate_url function.

    Function opens the directory containing topic txt files (which contain the result set of URL's)
    :return:
    """
    categories = config.get_categories()
    for category in categories:
        category = category.split('.')[0]
        print(category)

        topics_directory_path = os.path.join(config.get_app_root(), 'urls', category)

        create_score_file(category)

        # walk the directory and find txt files
        for root, dirs, files in os.walk(topics_directory_path):
            for filename in files:
                if '.txt' in filename:
                    rank = 0
                    with open(os.path.join(config.get_app_root(), 'urls', category, filename), 'r') as url_file:
                        for url in url_file:
                            rank = rank + 1
                            print("filename: ", filename, " url: ", url, " rank: ", rank)
                            try:
                                url = url.rstrip()
                                topic = filename.rstrip('.txt')
                                # evaluate the URL using NewsScan
                                json_object = call_news_scan(config.get_newsscan_api() + url)
                                parse_json(json_object, category, topic, rank)
                            except NewsScanError as e:
                                logging.warning(f'{url.rstrip()} Could not be called by NewsScan')
                                print("Could not access ", url)
                                raise


def create_score_file(category):
    with open(os.path.join(config.get_app_root(), 'evaluation', category + '.csv'), 'a+', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            ["url", "topic", "rank", "readability", "sentence_level_sentiment", "sentence_level_objectivity", "bias",
             "credibility", "trust_metric", "google_page_rank", "alexa_reach_rank"])


# def evaluate(url, category, filename, rank):
#     """
#     This method is responsible for call
#     :param url:
#     :param category:
#     :param filename:
#     :param rank:
#     :return:
#     """
#     json_object = call_news_scan(config.get_newsscan_api() + url)
#     parse_json(json_object, category, filename, rank)


def call_news_scan(url):
    """Feed URL into NewsScan domain and port through an HTTP Request. Response will be a JSON object.

    :param url: URL of news article
    :return: JSON response
    """

    try:
        response = requests.get(url)
        json_object = response.json()
        return json_object
    except requests.exceptions.RequestException as e:
        logging.debug(f'Error accessing NewsScan due to {e}')
        raise
    except json.decoder.JSONDecodeError as e1:
        logging.debug(f'Error accessing NewsScan due to {e1}')
        raise
    except Exception as e2:
        logging.debug(f'Error accessing NewsScan due to {e2}')
        raise


def parse_json(j, category, topic, rank):
    """Parse JSON Object (response from NewsScan) - j - and save critical information to the appropriate category csv file.
    :param json_object:
    """

    with open(os.path.join(config.get_app_root(), 'evaluation', category + '.csv'), 'a+', newline='') as csv_file:
        if j['url'] is not None:
            url = j['url']

            #if j['nutrition']['readability']['status'] == 'ok':
            #    readability = j['nutrition']['readability']['main_score']
            if j.get('nutrition').get('readability').get('main_score') is not None:
                readability = round(j.get('nutrition').get('readability').get('main_score'), 2)
            else:
                readability = np.nan

            # if j['nutrition']['sentence_level_sentiment']['status'] == 'ok':
            #     sentence_level_sentiment = j['nutrition']['sentence_level_sentiment']['main_score']
            if j.get('nutrition').get('sentence_level_sentiment').get('main_score') is not None:
                sentence_level_sentiment = round(j.get('nutrition').get('sentence_level_sentiment').get('main_score'), 2)
            else:
                sentence_level_sentiment = np.nan

            # if j['nutrition']['sentence_level_objectivity']['status'] == 'ok':
            #     sentence_level_objectivity = j['nutrition']['sentence_level_objectivity']['main_score']
            if j.get('nutrition').get('sentence_level_objectivity').get('main_score') is not None:
                sentence_level_objectivity = round(j.get('nutrition').get('sentence_level_objectivity').get('main_score'), 2)
            else:
                sentence_level_objectivity = np.nan

            # if j['nutrition']['political bias']['status'] == 'ok':
            #     bias = j['nutrition']['political bias']['main_score']
            if j.get('nutrition').get('political bias').get('main_score') is not None:
                bias = round(j.get('nutrition').get('political bias').get('main_score'), 2)
            else:
                bias = np.nan

            # if j['nutrition']['credibility']['score'] != np.nan:
            if j.get('nutrition').get('credibility').get('score') is not None:
                # credibility = j['nutrition']['credibility']['score']
                credibility = round(j.get('nutrition').get('credibility').get('score'), 2)
            else:
                credibility = np.nan

            # if j['nutrition']['credibility']['Trust Metric'] != np.nan:
            #     trust_metric = j['nutrition']['credibility']['Trust Metric']
            if j.get('nutrition').get('credibility').get('Trust Metric') is not None:
                trust_metric = round(j.get('nutrition').get('credibility').get('Trust Metric'), 2)
            else:
                trust_metric = np.nan

            # if j['nutrition']['credibility']['Google PageRank'] != np.nan:
            #     google_page_rank = j['nutrition']['credibility']['Google PageRank']
            if j.get('nutrition').get('credibility').get('Google PageRank') is not None:
                google_page_rank = round(j.get('nutrition').get('credibility').get('Google PageRank'), 2)
            else:
                google_page_rank = np.nan

            # if j['nutrition']['credibility']['Alexa Reach Rank'] != np.nan:
            #     alexa_reach_rank = j['nutrition']['credibility']['Alexa Reach Rank']
            if j.get('nutrition').get('credibility').get('Alexa Reach Rank') is not None:
                alexa_reach_rank = round(j.get('nutrition').get('credibility').get('Alexa Reach Rank'), 2)
            else:
                alexa_reach_rank = np.nan

            print(readability, sentence_level_sentiment, sentence_level_objectivity, bias, credibility,
                  trust_metric, google_page_rank, alexa_reach_rank)

            # Write variables to csv_file
            writer = csv.writer(csv_file)
            writer.writerow([rank, url, topic, readability, sentence_level_sentiment, sentence_level_objectivity, bias,
                             credibility, trust_metric, google_page_rank, alexa_reach_rank])


if __name__ == '__main__':
    feed_news_scan()

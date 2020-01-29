"""
Author: Rebecca Marsh
This script is used for extracting online news results from a list of queries and
copying their urls into a text file
"""

import os
from newsapi import NewsApiClient


def get_app_root():
    """Get path of project root"""
    path_of_script = os.path.dirname(os.path.realpath(__file__))
    return os.path.abspath(os.path.join(path_of_script, '..'))


def get_urls(topics_string):
    """Open file with list of topics. Read each topic and call google news api to request articles ranked relevant to
    this topic.  Write the result set of urls to a new file (one file per topic). With 20 topics and 2 categories, 40
    files will be created.
    :return None"""

    news_api = NewsApiClient(api_key='a68f4ec372ce4a6bae16c4bb7cd832fe')
    topics_file_path = os.path.join(get_app_root(), 'topics', topics_string)

    with open(topics_file_path, 'r') as topics_file:
        for topic in topics_file:
            # remove new line character from topic string
            topic = topic.rstrip()

            # Replace apostrophe's and capitalize each word in topic (for title url file)
            capitalized_topic = (topic.replace('\'', '')).title()

            # Create a new file to save result set
            urls_file_path = os.path.join(get_app_root(), 'urls', topics_string.split('.')[0],
                                          (capitalized_topic + '.txt').replace(' ', ''))
            # Write result set of urls to new file
            with open(urls_file_path, 'w') as urls_file:
                all_articles = news_api.get_everything(q=topic, sort_by='relevancy', page_size=100)
                for article in all_articles['articles']:
                    urls_file.write(article['url'])
                    urls_file.write('\n')


if __name__ == '__main__':
    category_list = ['politics_topics.txt', 'economics_topics.txt']
    for category in category_list:
        get_urls(category)






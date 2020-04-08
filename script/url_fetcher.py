"""
Author: Rebecca Marsh
This script is used for extracting online news results from a list of queries and
copying their urls into a text file
"""

import os
from newsapi import NewsApiClient
import config


def format_file_name(topic):
    """
    :param topic: individual topic from category
    :return: a file name string in camel case formatting without apostrophes or dashes, with 'txt' file ending
    """
    capitalized_topic = topic.replace('\'', '').title()
    return (capitalized_topic + '.txt').replace(' ', '').replace('-', '')


def get_urls(category):
    """Open file with list of topics. Read each topic and call google news api to request articles ranked relevant to
    this topic.  Write the result set of urls to a new file (one file per topic). With 20 topics and 2 categories, 40
    files will be created.
    The variable content_length copies the length of the article in the result.  The first 274 characters are printed in
    the output text and the rest of the article is included under article['content'] in the format '[+n chars]'.
    In order to get this length, the content string is converted into a list containing words, and the last element
    contains the length [+n chars]'.
    :param category:
    """
    news_api = NewsApiClient(api_key=config.get_newsapi_key())
    topics_list_file_path = os.path.join(config.get_app_root(), 'topics', category)

    with open(topics_list_file_path, 'r') as topics_file:
        for topic in topics_file:
            # remove new line character from topic string
            topic = topic.rstrip()
            print(topic)
            topic_file_name = format_file_name(topic)

            # Create a new file to save result set
            urls_file_path = os.path.join(config.get_app_root(), 'urls', category.split('.')[0], topic_file_name)
            # Write result set of urls to new file
            with open(urls_file_path, 'w') as urls_file:
                all_articles = news_api.get_everything(q=topic, sort_by='relevancy', page_size=100)
                for article in all_articles['articles']:
                    # Get length of the article and url
                    if article['content'] is not None:
                        if len(article['content']) < 274:
                            content_length = len(article['content'])
                        else:
                            content = article['content'].split()
                            content_length = int(content[-2].replace('[', '').replace('+', '')) + 274
                    print(article['url'], content_length)
                    urls_file.write(article['url'])
                    urls_file.write(', ')
                    urls_file.write(str(content_length))
                    urls_file.write('\n')


if __name__ == '__main__':
    category_list = config.get_categories()
    for category in category_list:
        get_urls(category)






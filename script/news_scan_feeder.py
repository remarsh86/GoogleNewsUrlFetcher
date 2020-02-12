"""
The purpose of this function(?)/script is:
1. For each topic, feed each fetched url into NewsScan Software through the NewsScan domain and port:
http://ariadne.is.inf.uni-due.de:7999/nutrition?
2. Collect results (in json format) for each call to NewsScanner and save results to a csv file

The resulting csv file will contain 100 rows, each row containing the resulting NewsScan scores for a url. The results
for each topic will be saved in one csv file.


"""
import os
import config


def open_url_file():
    """
    The following opens the topics files containing the result set from Google News for each topic and feeds the URLS
    into the evaluate_url function.
    :return:
    """
    categories = config.get_categories()
    for category in categories:
        category = category.split('.')[0]
        topics_file_path = os.path.join(config.get_app_root(), 'urls', category)

        # walk the directory and find txt files
        for root, dirs, files in os.walk(topics_file_path):
            for filename in files:
                if '.txt' in filename:
                    with open(os.path.join(config.get_app_root(), 'urls', category, filename), 'r') as url_file:
                        for url in url_file:
                            evaluate_url(url)


def evaluate_url(url):
    """Feed URL into NewsScan domain and port

    :param url: URL of news article
    :return:
    """
    pass


if __name__ == '__main__':
    open_url_file()

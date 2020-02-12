"""
This file contains config settings and functions
"""
import os

CATEGORY_LIST = ['politics_topics.txt', 'economics_topics.txt']


def get_categories():
    return CATEGORY_LIST


def get_app_root():
    """
    Get path of project root
    :return:
    """
    return os.path.dirname(os.path.realpath(__file__))

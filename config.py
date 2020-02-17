"""
This file contains config settings and functions
"""
import os

CATEGORY_LIST = ['politics_topics.txt', 'economics_topics.txt']
NEWSSCAN_CALL = "http://ariadne.is.inf.uni-due.de:7999/nutrition?url="


def get_categories():
    return CATEGORY_LIST


def get_app_root():
    """
    Get path of project root
    :return:
    """
    return os.path.dirname(os.path.realpath(__file__))


def get_newsscan_api():
    """
    Get string for calling NewsScan
    :return: returns a string for calling NewsScan
    """
    return NEWSSCAN_CALL

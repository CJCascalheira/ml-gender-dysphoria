"""
Download Reddit data from Pushshift
"""

# Import packages - crawl functions
import os

import requests
import time

# Import packages - data manipulation
import datetime as dt
import pandas as pd

# Set working directory
my_path = os.getcwd()

# Set the URL
url = "https://api.pushshift.io/reddit/search/submission"

# region DEFINE FUNCTIONS

# Define crawl page function - Alex Patry
def crawl_page(subreddit: str, last_page = None):
  """
  Crawl a page of results from a given subreddit.

  :param subreddit: The subreddit to crawl.
  :param last_page: The last downloaded page.

  :return: A page or results.
  """
  params = {"subreddit": subreddit, "size": 500, "sort": "desc", "sort_type": "created_utc"}
  if last_page is not None:
    if len(last_page) > 0:
      # resume from where we left at the last page
      params["before"] = last_page[-1]["created_utc"]
    else:
      # the last page was empty, we are past the last page
      return []
  results = requests.get(url, params)
  if not results.ok:
    # something wrong happened
    raise Exception("Server returned status code {}".format(results.status_code))
  return results.json()["data"]

# Define crawl reddit function - Alex Patry
def crawl_subreddit(subreddit, max_submissions = 10000):
  """
  Crawl submissions from a subreddit.

  :param subreddit: The subreddit to crawl.
  :param max_submissions: The maximum number of submissions to download.

  :return: A list of submissions.
  """
  submissions = []
  last_page = None
  while last_page != [] and len(submissions) < max_submissions:
    last_page = crawl_page(subreddit, last_page)
    submissions += last_page
    time.sleep(5)
  return submissions[:max_submissions]

# endregion

# region DOWNLOAD FROM SUBREDDIT - TRANS

# Pull all the submissions from r/trans
submissions_trans = crawl_subreddit("trans")

# Initialize empty lists
record_id = []
sub_reddit = []
text = []
post_time = []
n_comments = []

# Loop to assign values to lists
for submission in submissions_trans:
    record_id.append(submission['id'])
    post_time.append(submission['created_utc'])
    sub_reddit.append(submission['subreddit'])
    # Avoid the key error for a missing selftext key
    try:
        text.append(submission['selftext'])
    except KeyError:
        text.append('')
    n_comments.append(submission['num_comments'])

# Initialize empty data frame
df_trans = pd.DataFrame()

# Create data frame from lists
df_trans['record_id'] = record_id
df_trans['post_time'] = post_time
df_trans['subreddit'] = sub_reddit
df_trans['text'] = text
df_trans['n_comments'] = n_comments

# Empty list initialization
new_time = []

# Create a datetime object
for i in range(0, len(df_trans)):
     new_time.append(dt.datetime.fromtimestamp(df_trans['post_time'][i]))

# Overwrite the timestamp with the datetime object
df_trans['post_time'] = new_time

# Write to file
df_trans.to_csv(my_path + '\data\\raw\\pushshift\\reddit_trans_2022.csv')

# endregion

# region DOWNLOAD FROM SUBREDDIT - FTM

# Pull all the submissions from r/ftm
submissions_ftm = crawl_subreddit("ftm")

# Initialize empty lists
record_id = []
sub_reddit = []
text = []
post_time = []
n_comments = []

# Loop to assign values to lists
for submission in submissions_ftm:
    record_id.append(submission['id'])
    post_time.append(submission['created_utc'])
    sub_reddit.append(submission['subreddit'])
    # Avoid the key error for a missing selftext key
    try:
        text.append(submission['selftext'])
    except KeyError:
        text.append('')
    n_comments.append(submission['num_comments'])

# Initialize empty data frame
df_ftm = pd.DataFrame()

# Create data frame from lists
df_ftm['record_id'] = record_id
df_ftm['post_time'] = post_time
df_ftm['subreddit'] = sub_reddit
df_ftm['text'] = text
df_ftm['n_comments'] = n_comments

# Empty list initialization
new_time = []

# Create a datetime object
for i in range(0, len(df_ftm)):
     new_time.append(dt.datetime.fromtimestamp(df_ftm['post_time'][i]))

# Overwrite the timestamp with the datetime object
df_ftm['post_time'] = new_time

# Write to file
df_ftm.to_csv(my_path + '\data\\raw\\pushshift\\reddit_ftm_2022.csv')

# endregion

# region DOWNLOAD FROM SUBREDDIT - MTF

# Pull all the submissions from r/mtf
submissions_mtf = crawl_subreddit("MtF")

# Initialize empty lists
record_id = []
sub_reddit = []
text = []
post_time = []
n_comments = []

# Loop to assign values to lists
for submission in submissions_mtf:
    record_id.append(submission['id'])
    post_time.append(submission['created_utc'])
    sub_reddit.append(submission['subreddit'])
    # Avoid the key error for a missing selftext key
    try:
        text.append(submission['selftext'])
    except KeyError:
        text.append('')
    n_comments.append(submission['num_comments'])

# Initialize empty data frame
df_mtf = pd.DataFrame()

# Create data frame from lists
df_mtf['record_id'] = record_id
df_mtf['post_time'] = post_time
df_mtf['subreddit'] = sub_reddit
df_mtf['text'] = text
df_mtf['n_comments'] = n_comments

# Empty list initialization
new_time = []

# Create a datetime object
for i in range(0, len(df_mtf)):
     new_time.append(dt.datetime.fromtimestamp(df_mtf['post_time'][i]))

# Overwrite the timestamp with the datetime object
df_mtf['post_time'] = new_time

# Write to file
df_mtf.to_csv(my_path + '\data\\raw\\pushshift\\reddit_mtf_2022.csv')

# endregion

# region DOWNLOAD FROM SUBREDDIT - NONBINARY

# Pull all the submissions from r/NonBinary
submissions_nonbinary = crawl_subreddit("NonBinary")

# Initialize empty lists
record_id = []
sub_reddit = []
text = []
post_time = []
n_comments = []

# Loop to assign values to lists
for submission in submissions_nonbinary:
    record_id.append(submission['id'])
    post_time.append(submission['created_utc'])
    sub_reddit.append(submission['subreddit'])
    # Avoid the key error for a missing selftext key
    try:
        text.append(submission['selftext'])
    except KeyError:
        text.append('')
    n_comments.append(submission['num_comments'])

# Initialize empty data frame
df_nonbinary = pd.DataFrame()

# Create data frame from lists
df_nonbinary['record_id'] = record_id
df_nonbinary['post_time'] = post_time
df_nonbinary['subreddit'] = sub_reddit
df_nonbinary['text'] = text
df_nonbinary['n_comments'] = n_comments

# Empty list initialization
new_time = []

# Create a datetime object
for i in range(0, len(df_nonbinary)):
     new_time.append(dt.datetime.fromtimestamp(df_nonbinary['post_time'][i]))

# Overwrite the timestamp with the datetime object
df_nonbinary['post_time'] = new_time

# Write to file
df_nonbinary.to_csv(my_path + '\data\\raw\\pushshift\\reddit_nonbinary_2022.csv')

# endregion

# region DOWNLOAD FROM SUBREDDIT - GENDERDYSPHORIA

# Pull all the submissions from r/GenderDysphoria
submissions_dysphoria = crawl_subreddit("GenderDysphoria")

# Initialize empty lists
record_id = []
sub_reddit = []
text = []
title = []
post_time = []
n_comments = []

# Loop to assign values to lists
for submission in submissions_dysphoria:
    record_id.append(submission['id'])
    post_time.append(submission['created_utc'])
    sub_reddit.append(submission['subreddit'])
    # Avoid the key error for a missing selftext key
    try:
        text.append(submission['selftext'])
    except KeyError:
        text.append('')
    title.append(submission['title'])
    n_comments.append(submission['num_comments'])

# Initialize empty data frame
df_dysphoria = pd.DataFrame()

# Create data frame from lists
df_dysphoria['record_id'] = record_id
df_dysphoria['post_time'] = post_time
df_dysphoria['subreddit'] = sub_reddit
df_dysphoria['text'] = text
df_dysphoria['title'] = title
df_dysphoria['n_comments'] = n_comments

# Empty list initialization
new_time = []

# Create a datetime object
for i in range(0, len(df_dysphoria)):
     new_time.append(dt.datetime.fromtimestamp(df_dysphoria['post_time'][i]))

# Overwrite the timestamp with the datetime object
df_dysphoria['post_time'] = new_time

# Write to file
df_dysphoria.to_csv(my_path + '\data\\raw\\reddit_dysphoria\\reddit_dysphoria.csv')

# endregion

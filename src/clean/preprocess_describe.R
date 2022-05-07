# Dependencies
library(tidyverse)
library(lubridate)

# Import
df_truth_raw <- read_csv("data/raw/df_truth_raw.csv")

# Import Big Query trans files
my_csvs <- list.files("data/raw/google_bigquery_trans/")
my_files1 <- paste0("data/raw/google_bigquery_trans/", my_csvs)
files_list <- lapply(my_files1, read_csv)

# Combine data frames
df_bq_trans <- bind_rows(files_list, .id = "column_label") %>%
  select(-column_label)

# Import Pushshift trans files
my_csvs <- list.files("data/raw/pushshift/")
my_files1 <- paste0("data/raw/pushshift/", my_csvs)
files_list <- lapply(my_files1, read_csv)

# Combine data frames
df_ps_trans <- bind_rows(files_list, .id = "column_label") %>%
  select(-column_label)

# Import gender dysphoria subreddit
df_dysphoria <- read_csv("data/cleaned/dysphoria_reddit/df_truth_dysphoria_coded.csv")

# Set seed
set.seed(1234567)

# BALANCE DATA: TRAINING DATA ---------------------------------------------

# Create the training set
df_truth_dysphoria <- df_dysphoria %>%
  # Combine with manually coded data
  bind_rows(df_truth_raw[, c("text", "dysphoria")])
  
# Keep positive examples
df_truth_dysphoria_pos <- df_truth_dysphoria %>%
  filter(dysphoria == 1)

# Count the number of positive examples
n_pos <- df_truth_dysphoria_pos %>%
  nrow()

# Balance the data
df_truth_raw1 <- df_truth_dysphoria %>%
  filter(dysphoria == 0) %>%
  sample_n(n_pos) %>%
  bind_rows(df_truth_dysphoria_pos)

# Shuffle the data
df_truth_raw2 <- df_truth_raw1[sample(1:nrow(df_truth_raw1)), ]

# COMBINE DATA: TESTING DATA ----------------------------------------------

# Clean the data from Big Query 
df_bq_trans1 <- df_bq_trans %>%
  # Create datetime
  mutate(post_time = as_datetime(created_utc)) %>%
  # Select columns
  select(id, selftext, title, post_time) %>%
  # Remove reddit-specific nonsense
  filter(!selftext %in% c("[deleted]", "[removed]")) %>%
  # Unite the title and text columns
  unite(text, c(title, selftext), sep = " ") %>%
  # Remove missing
  filter(!is.na(text)) %>%
  # Add no label
  mutate(dysphoria = rep(NA_character_, nrow(.))) %>%
  # Select the columns for merging
  select(text, post_time, dysphoria) 

# Clean data from Pushshift and combine with data from Big Query
df_primary_raw <- df_ps_trans %>%
  # Select columns
  select(text, post_time) %>%
  # Remove reddit-specific nonsense
  filter(!text %in% c("[deleted]", "[removed]")) %>%
  # Remove missing
  filter(!is.na(text)) %>%
  # Add no label
  mutate(dysphoria = rep(NA_character_, nrow(.))) %>%
  # Combine with Big Query
  bind_rows(df_bq_trans1) %>%
  mutate(
    # Remove posts with fewer than 100 characters
    str_len = str_length(text)
  ) %>%
  filter(str_len > 100) %>%
  select(-str_len)

# CLEAN TRAINING DATA -----------------------------------------------------

# Clean up test data
df_truth <- df_truth_raw2 %>%
  # Remove links / URLs
  mutate(text = str_remove_all(text, " ?(f|ht)tp(s?)://(.*)[.][a-z]+")) %>%
  # Remove markdown links
  mutate(text = str_remove_all(text, "\\[.*\\]\\(.*\\)")) %>%
  # Replace whitespace characters
  mutate(text = str_replace_all(text, "\r\n\r\n", " ")) %>%
  mutate(text = str_replace_all(text, "\n", " ")) %>%
  # Remove strange characters
  mutate(text = str_remove_all(text, "&amp;#x200B;|â€¦|&lt;|&gt;|â€œ|ðŸ¥´|ðŸ¥²|â„¢|ðŸ¤·â€|â™€ï¸|â€™|â€|&gt;|Ã©||ðŸ™|ðŸŒˆ|ðŸ")) %>%
  # Recode characters
  mutate(text = recode(text, "&amp;" = "and", "Â´" = "'", "â€™" = "'")) %>%
  # Lowercase
  mutate(text = str_to_lower(text)) %>%
  # Remove missing
  mutate(
    # Remove posts with no characters
    str_len = str_length(text)
  ) %>%
  filter(str_len > 10) %>%
  select(-str_len)
df_truth

# Check for empty strings
df_truth %>%
  mutate(str_len = str_length(text)) %>%
  arrange(str_len)

# Get example of each class
df_truth %>%
  group_by(dysphoria) %>%
  sample_n(size = 2) %>%
  write_csv("data/results/ground_truth_examples.csv")

# CLEAN TESTING DATA ------------------------------------------------------

# Clean up text data
df_primary <- df_primary_raw %>%
  # Remove links / URLs
  mutate(text = str_remove_all(text, " ?(f|ht)tp(s?)://(.*)[.][a-z]+")) %>%
  # Remove markdown links
  mutate(text = str_remove_all(text, "\\[.*\\]\\(.*\\)")) %>%
  # Replace whitespace characters
  mutate(text = str_replace_all(text, "\r\n\r\n", " ")) %>%
  mutate(text = str_replace_all(text, "\n", " ")) %>%
  # Remove strange characters
  mutate(text = str_remove_all(text, "&amp;#x200B;|â€¦|&lt;|&gt;|â€œ|ðŸ¥´|ðŸ¥²|â„¢|ðŸ¤·â€|â™€ï¸|â€™|â€|&gt;|Ã©||ðŸ™|ðŸŒˆ|ðŸ")) %>%
  # Recode characters
  mutate(text = recode(text, "&amp;" = "and", "Â´" = "'", "â€™" = "'")) %>%
  # Lowercase
  mutate(text = str_to_lower(text)) %>%
  # Remove missing
  mutate(
    # Remove posts with no characters
    str_len = str_length(text)
  ) %>%
  filter(str_len > 10) %>%
  select(-str_len)
df_primary

# Check for empty strings
df_primary %>%
  mutate(str_len = str_length(text)) %>%
  arrange(str_len)

# Clean up part 2
df_primary_1 <- df_primary %>%
  mutate(text = str_trim(text)) %>%
  # Remove empty strings
  filter(text != " ", text != "") %>%
  # Keep distinct Reddit posts
  distinct(text, .keep_all = TRUE)

# Check for empty strings
df_primary_2 <- df_primary_1 %>%
  mutate(str_len = str_length(text)) %>%
  arrange(str_len) %>%
  # Keep Reddit posts with string lengths > 2
  filter(str_len > 2) %>%
  select(-str_len) 

# DESCRIBE DATASETS -------------------------------------------------------

# Since did not track participants, estimate with ratio
unique_participants <- distinct(df_bq_trans, author) %>% nrow()
total_participants <- nrow(df_bq_trans)
participant_ratio <- unique_participants / total_participants

# Ground truth - estimate number of participants
n_participant_truth <- nrow(df_truth_dysphoria) * participant_ratio

# Primary data - estimate number of participants
n_participant_primary <- (nrow(df_bq_trans) + nrow(df_ps_trans)) * participant_ratio

# Total estimate participants
n_participant_truth + n_participant_primary

# Total number of raw posts downloaded
nrow(df_truth_dysphoria) + nrow(df_bq_trans) + nrow(df_ps_trans)

# Count total posts after all cleaning
total_primary <- read_csv("data/cleaned/with_features/df_primary.csv")
total_truth <- read_csv("data/cleaned/with_features/df_truth.csv")
nrow(total_primary) + nrow(total_truth)

# Get times of primary data set 
df_primary_2 %>%
  # Filter the text in the final data set
  filter(text %in% total_primary$text) %>%
  # convert to hear
  mutate(year = year(post_time)) %>%
  # Count the number of posts per years
  count(year)

# WRITE TO FILE -----------------------------------------------------------

# Ground truth data set
write_csv(df_truth, "data/cleaned/df_truth_clean.csv")

# primary data set
df_primary_2 %>%
  select(text, dysphoria) %>% 
  write_csv("data/cleaned/df_primary_clean.csv")

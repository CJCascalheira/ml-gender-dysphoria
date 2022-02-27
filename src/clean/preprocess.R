# Dependencies
library(tidyverse)

# Import
df_train_raw <- read_csv("data/raw/df_train_raw.csv")

# Import Big Query trans files
my_csvs <- list.files("data/raw/google_bigquery_trans/")
my_files1 <- paste0("data/raw/google_bigquery_trans/", my_csvs)
files_list <- lapply(my_files1, read_csv)

# Combine data frames
df_bq_trans <- bind_rows(files_list, .id = "column_label") %>%
  select(-column_label)

# Import Big Query askscience files
my_csvs <- list.files("data/raw/google_bigquery_science/")
my_files1 <- paste0("data/raw/google_bigquery_science/", my_csvs)
files_list <- lapply(my_files1, read_csv)

# Combine data frames
df_science <- bind_rows(files_list, .id = "column_label") %>%
  select(-column_label)

# Import Pushshift trans files
my_csvs <- list.files("data/raw/pushshift/")
my_files1 <- paste0("data/raw/pushshift/", my_csvs)
files_list <- lapply(my_files1, read_csv)

# Combine data frames
df_ps_trans <- bind_rows(files_list, .id = "column_label") %>%
  select(-column_label)

# Import gender dysphoria subreddit
df_dysphoria <- read_csv("data/raw/reddit_dysphoria/reddit_dysphoria.csv")

# Set seed
set.seed(1234567)

# COMBINE DATA: TRAINING DATA ---------------------------------------------

# Create the training set
df_train_dysphoria <- df_dysphoria %>%
  # Remove missing
  filter(!is.na(text)) %>%
  # Remove reddit-specific nonsense
  filter(text != "[deleted]", text != "[removed]") %>%
  rename(selftext = text) %>%
  # Add positive label for all posts in Gender Dysphoria subreddit
  mutate(dysphoria = rep(1, nrow(.))) %>%
  # Unite the title and text columns
  unite(text, c(title, selftext), sep = " ") %>%
  # Select the columns for merging
  select(text, dysphoria) %>%
  # Combine with manually coded data
  bind_rows(df_train_raw[, c("text", "dysphoria")]) %>%
  mutate(
    # Remove unhelpful labels
    remove = if_else(str_detect(text, regex("selling|research|need mods|looking for mods|discord server|pm.*discord",
                                            ignore_case = TRUE)), 1, 0),
    # Remove posts with fewer than 100 characters
    str_len = str_length(text)
  ) %>%
  filter(remove == 0, str_len > 100) %>%
  select(-remove, -str_len)
  
# Count the number of positive examples
n_pos <- df_train_dysphoria %>%
  filter(dysphoria == 1) %>%
  nrow()

# Count the number of negative examples
n_neg <- nrow(df_train_dysphoria) - n_pos

# Total needed to supllement dataframe
n_sup <- n_pos - n_neg

# Clean up the science data
df_science1 <- df_science %>%
  distinct(selftext, .keep_all = TRUE) %>%
  select(text = selftext) %>%
  # Remove missing
  filter(!is.na(text)) %>%
  # Remove reddit-specific nonsense
  filter(text != "[deleted]", text != "[removed]")

# Balance the dataset with science data
df_train_raw1 <- df_science1[1:n_sup, ] %>%
  # Add negative labels to all science posts
  mutate(dysphoria = rep(0, nrow(.))) %>%
  # Combine with dysphoria dataframe
  bind_rows(df_train_dysphoria)

# Shuffle the data
df_train_raw2 <- df_train_raw1[sample(1:nrow(df_train_raw1)), ]

# Is the dataset balanced?
df_train_raw2 %>%
  count(dysphoria)

# COMBINE DATA: TESTING DATA ----------------------------------------------

# Clean the data from Big Query 
df_bq_trans1 <- df_bq_trans %>%
  # Remove missing
  filter(!is.na(selftext)) %>%
  # Remove reddit-specific nonsense
  filter(selftext != "[deleted]", selftext != "[removed]") %>%
  # Add no label
  mutate(dysphoria = rep(NA_character_, nrow(.))) %>%
  # Unite the title and text columns
  unite(text, c(title, selftext), sep = " ") %>%
  # Select the columns for merging
  select(text, dysphoria) 

# Clean data from Pushshift and combine with data from Big Query
df_test_raw <- df_ps_trans[, "text"] %>%
  # Remove missing
  filter(!is.na(text)) %>%
  # Remove reddit-specific nonsense
  filter(text != "[deleted]", text != "[removed]") %>%
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
df_train <- df_train_raw2 %>%
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
df_train

# Check for empty strings
df_train %>%
  mutate(str_len = str_length(text)) %>%
  arrange(str_len)

# CLEAN TESTING DATA ------------------------------------------------------

# Clean up text data
df_test <- df_test_raw %>%
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
df_test

# Check for empty strings
df_test %>%
  mutate(str_len = str_length(text)) %>%
  arrange(str_len)

# Clean up part 2
df_test_1 <- df_test %>%
  mutate(text = str_trim(text)) %>%
  # Remove empty strings
  filter(text != " ", text != "") %>%
  # Keep distinct Reddit posts
  distinct(text, .keep_all = TRUE)

# Check for empty strings
df_test_2 <- df_test_1 %>%
  mutate(str_len = str_length(text)) %>%
  arrange(str_len) %>%
  # Keep Reddit posts with string lengths > 2
  filter(str_len > 2) %>%
  select(-str_len) %>%
  mutate(temp_id = 1:nrow(.))

# Randomly select 100,000
test_sample <- sample(df_test_2$temp_id, 100000)

# Filter for sample
df_test_3 <- df_test_2 %>%
  filter(temp_id %in% test_sample) %>%
  select(-temp_id)

# WRITE TO FILE -----------------------------------------------------------

write_csv(df_train, "data/cleaned/df_train_clean.csv")
write_csv(df_test_3, "data/cleaned/df_test_clean.csv")

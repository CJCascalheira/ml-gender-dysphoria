# Dependencies
library(tidyverse)

# Import
df_truth_raw <- read_csv("data/raw/df_truth_raw.csv")

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
df_dysphoria <- read_csv("data/cleaned/dysphoria_reddit/df_truth_dysphoria_coded.csv")

# Set seed
set.seed(1234567)

# COMBINE DATA: TRAINING DATA ---------------------------------------------

# Clean up the dysphoria posts
df_dysphoria1 <- df_dysphoria
  # Remove who column
  # Remove posts that did not make the cut (i.e., coded as 0)

# Create the training set
df_truth_dysphoria <- df_dysphoria1 %>%
  # Combine with manually coded data
  bind_rows(df_truth_raw[, c("text", "dysphoria")])
  
# Count the number of positive examples
n_pos <- df_truth_dysphoria %>%
  filter(dysphoria == 1) %>%
  nrow()

# Count the number of negative examples
n_neg <- nrow(df_truth_dysphoria) - n_pos

# Total needed to supplement dataframe
n_sup <- n_pos - n_neg

# Clean up the science data
df_science1 <- df_science %>%
  distinct(selftext, .keep_all = TRUE) %>%
  # Select columns
  select(selftext, title) %>%
  # Remove reddit-specific nonsense
  filter(!selftext %in% c("[deleted]", "[removed]")) %>%
  # Unite the title and text columns
  unite(text, c(title, selftext), sep = " ") %>%
  # Remove missing
  filter(!is.na(text))

# Balance the dataset with science data
df_truth_raw1 <- df_science1[1:n_sup, ] %>%
  # Add negative labels to all science posts
  mutate(dysphoria = rep(0, nrow(.))) %>%
  # Combine with dysphoria dataframe
  bind_rows(df_truth_dysphoria)

# Shuffle the data
df_truth_raw2 <- df_truth_raw1[sample(1:nrow(df_truth_raw1)), ]

# Is the dataset balanced?
df_truth_raw2 %>%
  count(dysphoria)

# Get example of each class
df_truth_raw2 %>%
  group_by(dysphoria) %>%
  sample_n(size = 2) %>%
  write_csv("data/results/ground_truth_examples.csv")

# COMBINE DATA: TESTING DATA ----------------------------------------------

# Clean the data from Big Query 
df_bq_trans1 <- df_bq_trans %>%
  # Select columns
  select(id, selftext, title) %>%
  # Remove reddit-specific nonsense
  filter(!selftext %in% c("[deleted]", "[removed]")) %>%
  # Unite the title and text columns
  unite(text, c(title, selftext), sep = " ") %>%
  # Remove missing
  filter(!is.na(text)) %>%
  # Add no label
  mutate(dysphoria = rep(NA_character_, nrow(.))) %>%
  # Select the columns for merging
  select(text, dysphoria) 

# Clean data from Pushshift and combine with data from Big Query
df_primary_raw <- df_ps_trans[, "text"] %>%
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

# WRITE TO FILE -----------------------------------------------------------

write_csv(df_truth, "data/cleaned/df_truth_clean.csv")
write_csv(df_primary_2, "data/cleaned/df_primary_clean.csv")

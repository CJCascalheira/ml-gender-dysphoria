# Dependencies
library(tidyverse)

# Import BigQuery positive DASS files
my_csvs <- list.files("data/raw/dass/pos_examples/")
my_files1 <- paste0("data/raw/dass/pos_examples/", my_csvs)
files_list <- lapply(my_files1, read_csv)

# Add subreddit names 
for (i in 1:length(my_files1)) {
  files_list[[i]] <- files_list[[i]] %>%
    mutate(subreddit = rep(my_files1[i], nrow(.)))
}

# Combine data frames
df_pos <- bind_rows(files_list, .id = "column_label") %>%
  select(-column_label) %>%
  select(id, subreddit, everything()) %>%
  mutate(subreddit = str_extract(subreddit, regex("(?<=data/raw/dass/pos_examples/bigquery_)\\w*(?=_\\d*.csv)")))

# Import BigQuery negative DASS 9 subreddits
my_csvs <- list.files("data/raw/dass/neg_examples/")
my_files1 <- paste0("data/raw/dass/neg_examples/", my_csvs)
files_list <- lapply(my_files1, read_csv)

# Combine data frames
df_control <- bind_rows(files_list, .id = "column_label") %>%
  select(-column_label)

# Import BigQuery negative DASS 1 files
my_csvs <- list.files("data/raw/google_bigquery_science/")
my_files1 <- paste0("data/raw/google_bigquery_science/", my_csvs)
files_list <- lapply(my_files1, read_csv)

# Combine data frames
df_science <- bind_rows(files_list, .id = "column_label") %>%
  select(-column_label)

# Combined negative files
df_neg <- bind_rows(df_control, df_science) %>%
  # Select columns
  select(id, selftext, title) %>%
  # Remove reddit-specific nonsense
  filter(!selftext %in% c("[deleted]", "[removed]")) %>%
  # Unite the title and text columns
  unite(text, c(title, selftext), sep = " ") %>%
  # Remove missing
  filter(!is.na(text)) %>%
  # Remove missing
  mutate(
    # Remove posts with no characters
    str_len = str_length(text)
  ) %>%
  filter(str_len > 10) %>%
  select(-str_len) %>%
  sample_n(size = nrow(df_pos))

# Set seed
set.seed(1234567)

# CLEAN POSITIVE DASS EXAMPLES --------------------------------------------

# Preprocess
df_pos1 <- df_pos %>%
  # Select columns
  select(id, subreddit, selftext, title) %>%
  # Remove reddit-specific nonsense
  filter(!selftext %in% c("[deleted]", "[removed]")) %>%
  # Unite the title and text columns
  unite(text, c(title, selftext), sep = " ") %>%
  # Remove missing
  filter(!is.na(text)) %>%
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
  select(-str_len) %>%
  # Create label
  mutate(label = rep(1, nrow(.)))

# CLEAN NEGATIVE DASS EXAMPLES --------------------------------------------

# Preprocess
df_neg1 <- df_neg %>%
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
  # Create label
  mutate(label = rep(0, nrow(.)))

# SPLIT INTO DATASETS -----------------------------------------------------

# Divide positive examples based on subreddits
depression <- df_pos1 %>%
  filter(subreddit == "depression") %>%
  select(-subreddit) %>%
  # Sample to preserve memory
  slice_sample(n = 100000)

anxiety <- df_pos1 %>%
  filter(subreddit == "anxiety") %>%
  select(-subreddit) %>%
  # Sample to preserve memory
  slice_sample(n = 100000)

stress <- df_pos1 %>%
  filter(subreddit == "stress") %>%
  select(-subreddit)

suicide <- df_pos1 %>%
  filter(subreddit == "suicide") %>%
  select(-subreddit) %>%
  # Sample to preserve memory
  slice_sample(n = 100000)

# Combine positive and negative examples
depression_df <- df_neg1 %>%
  # Take a sample of the negative examples
  filter(id %in% sample(df_neg1$id, size = nrow(depression))) %>%
  bind_rows(depression)

anxiety_df <- df_neg1 %>%
  # Take a sample of the negative examples
  filter(id %in% sample(df_neg1$id, size = nrow(anxiety))) %>%
  bind_rows(anxiety)

stress_df <- df_neg1 %>%
  # Take a sample of the negative examples
  filter(id %in% sample(df_neg1$id, size = nrow(stress))) %>%
  bind_rows(stress)

suicide_df <- df_neg1 %>%
  # Take a sample of the negative examples
  filter(id %in% sample(df_neg1$id, size = nrow(suicide))) %>%
  bind_rows(suicide)

# Shuffle the dataframes
depression_df <- depression_df[sample(nrow(depression_df)), ]
anxiety_df <- anxiety_df[sample(nrow(anxiety_df)), ]
stress_df <- stress_df[sample(nrow(stress_df)), ]
suicide_df <- suicide_df[sample(nrow(suicide_df)), ]

# Check balance of class labels
count(depression_df, label)

# WRITE TO FILE -----------------------------------------------------------

write_csv(depression_df, "data/cleaned/dass/depression_df.csv")
write_csv(anxiety_df, "data/cleaned/dass/anxiety_df.csv")
write_csv(stress_df, "data/cleaned/dass/stress_df.csv")
write_csv(suicide_df, "data/cleaned/dass/suicide_df.csv")

# Dependencies
library(tidyverse)
library(tidytext)

# Import data
df_train <- read_csv("data/cleaned/liwc_results/df_train_liwc.csv") %>%
  rename(temp_id = `Source (A)`, text = `Source (B)`, dysphoria = `Source (C)`) %>%
  rename(function_liwc = `function`) %>%
  filter(temp_id != "temp_id") %>%
  mutate(dysphoria = as.integer(dysphoria))

df_test <- read_csv("data/cleaned/liwc_results/df_test_liwc.csv") %>%
  rename(temp_id = A, text = B, dysphoria = C) %>%
  rename(function_liwc = `function`) %>%
  filter(temp_id != "temp_id") %>%
  mutate(dysphoria = as.integer(dysphoria))

dsm5 <- read_csv("data/dsm5_gender_dysphoria.csv")

# Load stop words
data(stop_words)

# Get sentiments
afinn <- get_sentiments("afinn")

# Get slangSD: https://github.com/airtonbjunior/opinionMining/blob/master/dictionaries/slangSD.txt
slangsd <- read_delim("data/slangSD.txt", delim = "\t", col_names = FALSE) %>%
  rename(word = X1, value = X2)

# Combine sentiment libraries
sentiment_df <- bind_rows(afinn, slangsd) %>%
  distinct(word, .keep_all = TRUE)

# LIWC --------------------------------------------------------------------

# Count features
df_train %>%
  select(-temp_id, -text, -dysphoria) %>%
  names() %>%
  length()

# CLINICAL KEYWORDS -------------------------------------------------------

# Get clinical words
clinical_keywords <- dsm5 %>%
  unnest_tokens(word, dsm5) %>%
  filter(!(word %in% stop_words$word)) %>%
  count(word) %>%
  # Top words in the DSM-5
  arrange(desc(n)) %>%
  # Remove common words from top words
  filter(!(word %in% c("gender", "sex", "individuals", "children", "female"))) %>%
  # Select top 5 clinical keywords
  filter(n >= 25) %>%
  pull(word)
clinical_keywords

# Add features - train
df_train1 <- df_train %>%
  mutate(
    clinical_keywords = if_else(str_detect(text, regex("dysphor*|natal|disorder|assign*|develop*")), 1, 0)
  )

# Add features - test
df_test1 <- df_test %>%
  mutate(
    clinical_keywords = if_else(str_detect(text, regex("dysphor*|natal|disorder|assign*|develop*")), 1, 0)
  )

# SENTIMENT: VALENCE APPROACH ---------------------------------------------

# Add features - train
df_train2 <- df_train1 %>%
  # Reduce df size
  select(temp_id, text) %>%
  # Tokenize Reddit post
  unnest_tokens(word, text) %>%
  # Get sentiment of words
  left_join(sentiment_df) %>%
  # Recode missing to 0 sentiment
  mutate(value = if_else(is.na(value), 0, value)) %>%
  # Group by post
  group_by(temp_id) %>%
  # Calculate total sentiment of post
  summarize(sentiment_valence = sum(value)) %>%
  # Join to main dataframe
  left_join(df_train) %>%
  # Rearrange the variables
  select(temp_id, text, dysphoria, everything())
df_train2

# Add features - test
df_test2 <- df_test1 %>%
  # Reduce df size
  select(temp_id, text) %>%
  # Tokenize Reddit post
  unnest_tokens(word, text) %>%
  # Get sentiment of words
  left_join(sentiment_df) %>%
  # Recode missing to 0 sentiment
  mutate(value = if_else(is.na(value), 0, value)) %>%
  # Group by post
  group_by(temp_id) %>%
  # Calculate total sentiment of post
  summarize(sentiment_valence = sum(value)) %>%
  # Join to main dataframe
  left_join(df_test) %>%
  # Rearrange the variables
  select(temp_id, text, dysphoria, everything())
df_test2

# SAVE DATAFRAMES WITH FEATURES -------------------------------------------

write_csv(df_train2, "data/cleaned/with_features/df_train.csv")
write_csv(df_test2, "data/cleaned/with_features/df_test.csv")

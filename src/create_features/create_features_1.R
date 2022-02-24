# Dependencies
library(tidyverse)
library(tidytext)
library(widyr)

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

# TOP N-GRAMS -------------------------------------------------------------

# Prepare the data
train_merge <- df_train2 %>%
  select(temp_id, text) %>%
  mutate(set = rep("train", nrow(.)))

df_full <- df_test2 %>%
  select(temp_id, text) %>%
  mutate(set = rep("test", nrow(.))) %>%
  bind_rows(train_merge)

# Top unigrams
unigram_df <- df_full %>%
  # Generate unigrams
  unnest_tokens(word, text) %>%
  # Remove stop words
  anti_join(stop_words) %>%
  count(word) %>%
  arrange(desc(n)) %>%
  # Clean up based on remainign stop words
  mutate(
    stop_word = if_else(str_detect(word, regex("^im$|that's|i’m|it’s|you’re|don’t|dont|It|can’t|lt|he’s|she’s|i’ve|doesn’t|didn’t|isn’t|there’s|that'll|how’s|they’ll|it’ll|would've|we’ll|they’ve|shouldn’t|that’s|i’ll|they’re|aren’t|i’d|won’t|what’s|you’ve|we’re|wouldn’t|haven’t|wasn’t|y'all|let’s|here’s|who’s|you’ll|couldn’t|weren’t|hasn’t|we’ve|ain’t|you’d|y’all")), 1, 0) 
  ) %>%
  # Remove remaining stop words
  filter(stop_word == 0)

# Select top 500 unigrams
unigram_vector <- unigram_df[1:500, ]$word

# Top bigrams
bigram_df <- df_full %>%
  # Generate bigrams
  unnest_ngrams(bigram, text, n = 2) %>%
  # Create a group linking the bigrams
  mutate(group = 1:nrow(.)) %>%
  # Separate the bigrams into two columns
  separate(bigram, c("word1", "word2"))

# Pointwise mutual information
bigram_df %>%
  gather(key = "position", value = "words", -temp_id, -set, -group) %>%
  pairwise_pmi(item = words, feature = group, sort = TRUE)

# SAVE DATAFRAMES WITH FEATURES -------------------------------------------

write_csv(df_train2, "data/cleaned/with_features/df_train.csv")
write_csv(df_test2, "data/cleaned/with_features/df_test.csv")

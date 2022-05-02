# Dependencies
library(tidyverse)
library(tidytext)
library(textdata)
library(widyr)

# Import data
df_truth <- read_csv("data/cleaned/liwc_results/df_truth_liwc.csv") %>%
  rename(text = `Source (A)`, dysphoria = `Source (B)`) %>%
  rename(function_liwc = `function`) %>%
  mutate(dysphoria = as.integer(dysphoria)) %>%
  # Create a temporary ID
  mutate(temp_id = 1:nrow(.)) %>%
  select(temp_id, everything())

df_truth <- df_truth[-1, ]

df_primary <- read_csv("data/cleaned/liwc_results/df_primary_liwc.csv") %>%
  rename(text = A, dysphoria = B) %>%
  rename(function_liwc = `function`)

df_primary <- df_primary[-1, ]  %>%
  mutate(dysphoria = as.integer(dysphoria)) %>%
  # Create a temporary ID
  mutate(temp_id = 1:nrow(.)) %>%
  select(temp_id, everything())

# Import the DSM-5 text on gender dysphoria
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
df_truth %>%
  select(-text, -dysphoria) %>%
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
df_truth1 <- df_truth %>%
  mutate(
    clinical_keywords = if_else(str_detect(text, regex("dysphor*|natal|disorder|assign*|develop*")), 1, 0)
  )

# Add features - test
df_primary1 <- df_primary %>%
  mutate(
    clinical_keywords = if_else(str_detect(text, regex("dysphor*|natal|disorder|assign*|develop*")), 1, 0)
  )

# SENTIMENT: VALENCE APPROACH ---------------------------------------------

# Add features - train
df_truth2 <- df_truth1 %>%
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
  left_join(df_truth1) %>%
  # Rearrange the variables
  select(temp_id, text, dysphoria, clinical_keywords, everything())
df_truth2

# Add features - test
df_primary2 <- df_primary1 %>%
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
  left_join(df_primary1) %>%
  # Rearrange the variables
  select(temp_id, text, dysphoria, clinical_keywords, everything())
df_primary2

# TOP N-GRAMS -------------------------------------------------------------

# Top unigrams
unigram_df <- df_truth2 %>%
  # Select key columns
  select(temp_id, text, dysphoria) %>%
  # Generate unigrams
  unnest_tokens(word, text, drop = FALSE) %>%
  # Remove stop words
  anti_join(stop_words) %>%
  count(dysphoria, word) %>%
  arrange(desc(n)) %>%
  # Clean up based on remaining stop words
  mutate(
    stop_word = if_else(str_detect(word, regex("^im$|that's|i’m|it’s|you’re|don’t|dont|It|can’t|lt|he’s|she’s|i’ve|doesn’t|didn’t|isn’t|there’s|that'll|how’s|they’ll|it’ll|would've|we’ll|they’ve|shouldn’t|that’s|i’ll|they’re|aren’t|i’d|won’t|what’s|you’ve|we’re|wouldn’t|haven’t|wasn’t|y'all|let’s|here’s|who’s|you’ll|couldn’t|weren’t|hasn’t|we’ve|ain’t|you’d|y’all")), 1, 0) 
  ) %>%
  # Remove remaining stop words
  filter(stop_word == 0) %>%
  select(-stop_word)

# Term frequency, inverse document frequency (tf-idf)
unigram_df1 <- unigram_df %>%
  # Calculate tf-idf
  bind_tf_idf(word, dysphoria, n) %>%
  # Get top tf-idf of unigrams for dysphoria posts
  arrange(desc(tf_idf)) %>%
  filter(dysphoria == 1) %>%
  # Remove words based on closed inspection of unigrams
  mutate(remove = if_else(str_detect(word, regex("’s|’d|'s|	
’ve|\\d|monday|tuesday|wednesday|thursday|friday|saturday|sunday|lockdown|covid|grammatical|film|eh|could’ve|december|vehicle|paint|ness|bout|brown|animals|âˆ|weather|bike|maria|albeit|amd|matt|minecraft")), 1, 0)) %>%
  filter(remove == 0)

# Select top 250 unigrams
unigram_vector <- unigram_df1[1:250, ]$word

# Generate bigrams
bigram_df <- df_truth2 %>%
  # Select key columns
  select(temp_id, text, dysphoria) %>%
  unnest_ngrams(bigram, text, n = 2, drop = FALSE) %>%
  # Separate the bigrams into two columns
  separate(bigram, c("word1", "word2")) %>%
  # Remove stop words
  filter(!(word1 %in% stop_words$word)) %>%
  filter(!(word2 %in% stop_words$word)) %>%
  # Clean up based on remaining stop words
  mutate(
    stop_word1 = if_else(str_detect(word1, regex("^im$|that's|i’m|it’s|you’re|don’t|dont|It|can’t|lt|he’s|she’s|i’ve|doesn’t|didn’t|isn’t|there’s|that'll|how’s|they’ll|it’ll|would've|we’ll|they’ve|shouldn’t|that’s|i’ll|they’re|aren’t|i’d|won’t|what’s|you’ve|we’re|wouldn’t|haven’t|wasn’t|y'all|let’s|here’s|who’s|you’ll|couldn’t|weren’t|hasn’t|we’ve|ain’t|you’d|y’all")), 1, 0),
    stop_word2 = if_else(str_detect(word2, regex("^im$|that's|i’m|it’s|you’re|don’t|dont|It|can’t|lt|he’s|she’s|i’ve|doesn’t|didn’t|isn’t|there’s|that'll|how’s|they’ll|it’ll|would've|we’ll|they’ve|shouldn’t|that’s|i’ll|they’re|aren’t|i’d|won’t|what’s|you’ve|we’re|wouldn’t|haven’t|wasn’t|y'all|let’s|here’s|who’s|you’ll|couldn’t|weren’t|hasn’t|we’ve|ain’t|you’d|y’all")), 1, 0)
  ) %>%
  filter(stop_word1 == 0, stop_word2 == 0) %>%
  unite("bigram", c("word1", "word2"), sep = " ") %>%
  # Count top bigrams
  count(dysphoria, bigram) %>%
  arrange(desc(n))

# Term frequency, inverse document frequency (tf-idf)
bigram_df1 <- bigram_df %>%
  # Calculate tf-idf
  bind_tf_idf(bigram, dysphoria, n) %>%
  # Get top tf-idf of unigrams for dysphoria posts
  arrange(desc(tf_idf)) %>%
  filter(dysphoria == 1) %>%
  # Remove words based on closed inspection of unigrams
  mutate(remove = if_else(str_detect(bigram, regex("’s|’d|'s|	
’ve|\\d|monday|tuesday|wednesday|thursday|friday|saturday|sunday|lockdown|covid|^ive |^lot |minutes ago")), 1, 0)) %>%
  filter(remove == 0)

# Select top 250 bigrams
bigram_vector <- bigram_df1[1:250, ]$bigram

# Generate trigrams
trigram_df <- df_truth2 %>%
  # Select key columns
  select(temp_id, text, dysphoria) %>%
  unnest_ngrams(trigram, text, n = 3, drop = FALSE) %>%
  # Separate into three columns
  separate(trigram, c("word1", "word2", "word3"), sep = " ") %>%
  # Remove stop words
  filter(!word1 %in% stop_words$word,
         !word2 %in% stop_words$word,
         !word3 %in% stop_words$word) %>%
  # Clean up based on remaining stop words
  mutate(
    stop_word1 = if_else(str_detect(word1, regex("^im$|that's|i’m|it’s|you’re|don’t|dont|It|can’t|lt|he’s|she’s|i’ve|doesn’t|didn’t|isn’t|there’s|that'll|how’s|they’ll|it’ll|would've|we’ll|they’ve|shouldn’t|that’s|i’ll|they’re|aren’t|i’d|won’t|what’s|you’ve|we’re|wouldn’t|haven’t|wasn’t|y'all|let’s|here’s|who’s|you’ll|couldn’t|weren’t|hasn’t|we’ve|ain’t|you’d|y’all")), 1, 0),
    stop_word2 = if_else(str_detect(word2, regex("^im$|that's|i’m|it’s|you’re|don’t|dont|It|can’t|lt|he’s|she’s|i’ve|doesn’t|didn’t|isn’t|there’s|that'll|how’s|they’ll|it’ll|would've|we’ll|they’ve|shouldn’t|that’s|i’ll|they’re|aren’t|i’d|won’t|what’s|you’ve|we’re|wouldn’t|haven’t|wasn’t|y'all|let’s|here’s|who’s|you’ll|couldn’t|weren’t|hasn’t|we’ve|ain’t|you’d|y’all")), 1, 0) ,
    stop_word3 = if_else(str_detect(word3, regex("^im$|that's|i’m|it’s|you’re|don’t|dont|It|can’t|lt|he’s|she’s|i’ve|doesn’t|didn’t|isn’t|there’s|that'll|how’s|they’ll|it’ll|would've|we’ll|they’ve|shouldn’t|that’s|i’ll|they’re|aren’t|i’d|won’t|what’s|you’ve|we’re|wouldn’t|haven’t|wasn’t|y'all|let’s|here’s|who’s|you’ll|couldn’t|weren’t|hasn’t|we’ve|ain’t|you’d|y’all")), 1, 0) 
  ) %>%
  # Remove contracted stop words
  filter(
    stop_word1 == 0,
    stop_word2 == 0,
    stop_word3 == 0
  ) %>%
  # Combine into trigrams
  unite("trigram", c("word1", "word2", "word3"), sep = " ") %>%
  count(dysphoria, trigram) %>%
  arrange(desc(n))

# View trigrams to clean for internet nonsense
trigram_df1 <- trigram_df %>%
  # Manual remove of nonsense
  mutate(remove = if_else(str_detect(trigram, "\\d|ðÿ|^amp |amp | amp$|NA NA NA|poll$|jfe|_link|link_|playlist 3948ybuzmcysemitjmy9jg si|complete 3 surveys|gmail.com mailto:hellogoodbis42069 gmail.com|hellogoodbis42069 gmail.com mailto:hellogoodbis42069|comments 7n2i gay_marriage_debunked_in_2_minutes_obama_vs_alan|debatealtright comments 7n2i|gift card|amazon|action hirewheller csr|energy 106 fm|form sv_a3fnpplm8nszxfb width|â íœê í|âˆ âˆ âˆ"), 1, 0)) %>%
  filter(remove == 0) %>%
  # Calculate tf-idf
  bind_tf_idf(trigram, dysphoria, n) %>%
  # Get top tf-idf of unigrams for dysphoria posts
  arrange(desc(tf_idf)) %>%
  filter(dysphoria == 1)

# Select top 50 trigrams
trigram_vector <- trigram_df1[1:50, ]$trigram

# GROUND TRUTH DATASET

# Assign the unigrams as features
for (i in 1:length(unigram_vector)) {
  
  # Get the n-grams
  ngram <- unigram_vector[i]
  
  # Detect the n-gram with regular expressions
  x <- str_detect(df_truth2$text, regex(ngram))
  
  # Add the n-gram to the dataframe
  df_truth2[[ngram]] <- as.integer(x)  
}

# Assign the bigrams as features
for (i in 1:length(bigram_vector)) {
  
  # Get the n-grams
  ngram <- bigram_vector[i]
  
  # Detect the n-gram with regular expressions
  x <- str_detect(df_truth2$text, regex(ngram))
  
  # Add the n-gram to the dataframe
  df_truth2[[ngram]] <- as.integer(x)  
}

# Assign the trigrams as features
for (i in 1:length(trigram_vector)) {
  
  # Get the n-grams
  ngram <- trigram_vector[i]
  
  # Detect the n-gram with regular expressions
  x <- str_detect(df_truth2$text, regex(ngram))
  
  # Add the n-gram to the dataframe
  df_truth2[[ngram]] <- as.integer(x)  
}

# PRIMARY DATASET

# Assign the unigrams as features
for (i in 1:length(unigram_vector)) {
  
  # Get the n-grams
  ngram <- unigram_vector[i]
  
  # Detect the n-gram with regular expressions
  x <- str_detect(df_primary2$text, regex(ngram))
  
  # Add the n-gram to the dataframe
  df_primary2[[ngram]] <- as.integer(x)  
}

# Assign the bigrams as features
for (i in 1:length(bigram_vector)) {
  
  # Get the n-grams
  ngram <- bigram_vector[i]
  
  # Detect the n-gram with regular expressions
  x <- str_detect(df_primary2$text, regex(ngram))
  
  # Add the n-gram to the dataframe
  df_primary2[[ngram]] <- as.integer(x)  
}

# Assign the trigrams as features
for (i in 1:length(trigram_vector)) {
  
  # Get the n-grams
  ngram <- trigram_vector[i]
  
  # Detect the n-gram with regular expressions
  x <- str_detect(df_primary2$text, regex(ngram))
  
  # Add the n-gram to the dataframe
  df_primary2[[ngram]] <- as.integer(x)  
}

# SAVE DATAFRAMES WITH FEATURES -------------------------------------------

write_csv(df_truth2, "data/cleaned/features_temp/df_truth.csv")
write_csv(df_primary2, "data/cleaned/features_temp/df_primary.csv")

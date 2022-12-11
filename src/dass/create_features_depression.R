# Dependencies
library(tidyverse)
library(tidytext)
library(stopwords)

# Set a random seed
set.seed(1234567)

# Import data
depression_df <- read_csv("data/cleaned/dass/depression_df.csv")

# Load stop words
data(stop_words)

# GET FEATURES ------------------------------------------------------------

# Log likelihood ratio test
# http://uc-r.github.io/creating-text-features#likelihood
# https://web.stanford.edu/~jurafsky/slp3/3.pdf
# https://leimao.github.io/blog/Maximum-Likelihood-Estimation-Ngram/
# https://www.youtube.com/watch?v=UyC0bBiZY-A
# https://www.inf.ed.ac.uk/teaching/courses/fnlp/lectures/03_slides.pdf

#* GET TRIGRAMS -----------------------------------------------------------

# Generate and clean trigrams
trigrams_df <- depression_df %>%
  unnest_ngrams(trigram, text, n = 3) %>%
  # Words into individual columns
  separate(trigram, c("word1", "word2", "word3")) %>%
  # Remove stop words
  filter(!word1 %in% stop_words$word) %>%
  filter(!(word1 %in% stopwords(source = "snowball"))) %>%
  filter(!(word1 %in% stopwords(source = "stopwords-iso"))) %>%
  filter(!word2 %in% stop_words$word) %>%
  filter(!(word2 %in% stopwords(source = "snowball"))) %>%
  filter(!(word2 %in% stopwords(source = "stopwords-iso"))) %>%
  filter(!word3 %in% stop_words$word) %>%
  filter(!(word3 %in% stopwords(source = "snowball"))) %>%
  filter(!(word3 %in% stopwords(source = "stopwords-iso"))) %>%
  # Clean up based on remaining stop words
  mutate(
    stop_word1 = if_else(str_detect(word1, regex("^im$|that's|i’m|it’s|you’re|don’t|dont|It|can’t|lt|he’s|she’s|i’ve|doesn’t|didn’t|isn’t|there’s|that'll|how’s|they’ll|it’ll|would've|we’ll|they’ve|shouldn’t|that’s|i’ll|they’re|aren’t|i’d|won’t|what’s|you’ve|we’re|wouldn’t|haven’t|wasn’t|y'all|let’s|here’s|who’s|you’ll|couldn’t|weren’t|hasn’t|we’ve|ain’t|you’d|y’all")), 1, 0),
    stop_word2 = if_else(str_detect(word2, regex("^im$|that's|i’m|it’s|you’re|don’t|dont|It|can’t|lt|he’s|she’s|i’ve|doesn’t|didn’t|isn’t|there’s|that'll|how’s|they’ll|it’ll|would've|we’ll|they’ve|shouldn’t|that’s|i’ll|they’re|aren’t|i’d|won’t|what’s|you’ve|we’re|wouldn’t|haven’t|wasn’t|y'all|let’s|here’s|who’s|you’ll|couldn’t|weren’t|hasn’t|we’ve|ain’t|you’d|y’all")), 1, 0),
    stop_word3 = if_else(str_detect(word3, regex("^im$|that's|i’m|it’s|you’re|don’t|dont|It|can’t|lt|he’s|she’s|i’ve|doesn’t|didn’t|isn’t|there’s|that'll|how’s|they’ll|it’ll|would've|we’ll|they’ve|shouldn’t|that’s|i’ll|they’re|aren’t|i’d|won’t|what’s|you’ve|we’re|wouldn’t|haven’t|wasn’t|y'all|let’s|here’s|who’s|you’ll|couldn’t|weren’t|hasn’t|we’ve|ain’t|you’d|y’all")), 1, 0)
  ) %>%
  # Remove remaining stop words
  filter(stop_word1 == 0, stop_word2 == 0, stop_word3 == 0) %>%
  select(-stop_word1, -stop_word2, -stop_word3)

# Compute counts for individual words
count_w1 <- trigrams_df %>%
  count(label, word1)

count_w2 <- trigrams_df %>%
  count(label, word2)

count_w3 <- trigrams_df %>%
  count(label, word3)

# Compute counts for trigrams
count_w123 <- trigrams_df %>%
  count(label, word1, word2, word3)

# Compute counts for C given B
count_w12 <- trigrams_df %>%
  count(label, word1, word2)

# Original number of trigrams
N <- nrow(trigrams_df)

# Join the data and compute log-likelihood
trigrams_ll <- count_w123 %>%
  left_join(count_w1, by = "word1") %>%
  left_join(count_w2, by = "word2") %>%
  left_join(count_w3, by = "word3") %>%
  left_join(count_w12, by = c("word1", "word2")) %>%
  # Drop repeated columns
  select(-label.y, -label.x.x, -label.y.y, -label) %>%
  # Rename values
  rename(c_w123 = n.x, c_w1 = n.y, c_w2 = n.x.x, c_w3 = n.y.y, c_w12 = n, label = label.x) %>%
  # Compute probabilities and log-likelihood
  mutate(
    p_w3 = c_w3 / N,
    p_trigram = c_w123 / c_w12,
    p_not = (c_w3 - c_w123) / (N - c_w12),
    LL = log((pbinom(c_w123, c_w12, p_w3) * pbinom(c_w3 - c_w123, N - c_w12, p_w3)) / (pbinom(c_w123, c_w12, p_trigram) * pbinom(c_w3 - c_w123, N - c_w12, p_w3)))
) %>%
  arrange(desc(LL)) %>%
  # Select columns of interest
  select(label, word1, word2, word3, n = c_w123, LL) %>%
  # Combine words 
  unite("trigram", c("word1", "word2", "word3"), sep = " ") %>%
  # Keep the unique trigrams
  distinct(trigram, .keep_all = TRUE) %>%
  filter(label == 1)

# Calculate TF-IDF
trigrams_tf_idf <- trigrams_df %>% 
  # Combine the words again
  unite("trigram", c("word1", "word2", "word3"), sep = " ") %>%
  # Count and sort the ngrams
  count(label, trigram, sort = TRUE) %>%
  # Calculate tf-idf
  bind_tf_idf(trigram, label, n) %>%
  arrange(desc(tf_idf)) %>%
  filter(label == 1)

# Select trigrams and clean
trigrams <- left_join(trigrams_ll, trigrams_tf_idf) %>%
  arrange(desc(LL)) %>%
  # Clean the trigrams
  mutate(
    remove = if_else(str_detect(trigram, regex("\\d|^amp|amp$|tl dr")), 1, 0)
  ) %>%
  filter(remove == 0) %>%
  slice_head(n = 1666) %>%
  select(trigram)

# Save to file
write_csv(trigrams, "data/cleaned/dass/ngrams/trigrams_depression.csv")

#* GET BIGRAMS ------------------------------------------------------------

# Generate and clean bigrams
bigrams_df <- depression_df %>%
  unnest_ngrams(bigram, text, n = 2) %>%
  # Words into individual columns
  separate(bigram, c("word1", "word2")) %>%
  # Remove stop words
  filter(!word1 %in% stop_words$word) %>%
  filter(!(word1 %in% stopwords(source = "snowball"))) %>%
  filter(!(word1 %in% stopwords(source = "stopwords-iso"))) %>%
  filter(!word2 %in% stop_words$word) %>%
  filter(!(word2 %in% stopwords(source = "snowball"))) %>%
  filter(!(word2 %in% stopwords(source = "stopwords-iso"))) %>%
  # Clean up based on remaining stop words
  mutate(
    stop_word1 = if_else(str_detect(word1, regex("^im$|that's|i’m|it’s|you’re|don’t|dont|It|can’t|lt|he’s|she’s|i’ve|doesn’t|didn’t|isn’t|there’s|that'll|how’s|they’ll|it’ll|would've|we’ll|they’ve|shouldn’t|that’s|i’ll|they’re|aren’t|i’d|won’t|what’s|you’ve|we’re|wouldn’t|haven’t|wasn’t|y'all|let’s|here’s|who’s|you’ll|couldn’t|weren’t|hasn’t|we’ve|ain’t|you’d|y’all")), 1, 0),
    stop_word2 = if_else(str_detect(word2, regex("^im$|that's|i’m|it’s|you’re|don’t|dont|It|can’t|lt|he’s|she’s|i’ve|doesn’t|didn’t|isn’t|there’s|that'll|how’s|they’ll|it’ll|would've|we’ll|they’ve|shouldn’t|that’s|i’ll|they’re|aren’t|i’d|won’t|what’s|you’ve|we’re|wouldn’t|haven’t|wasn’t|y'all|let’s|here’s|who’s|you’ll|couldn’t|weren’t|hasn’t|we’ve|ain’t|you’d|y’all")), 1, 0)
  ) %>%
  # Remove remaining stop words
  filter(stop_word1 == 0, stop_word2 == 0) %>%
  select(-stop_word1, -stop_word2)

# Calculate TF-IDF
bigrams_tf_idf <- bigrams_df %>% 
  # Combine the words again
  unite("bigram", c("word1", "word2"), sep = " ") %>%
  # Count and sort the ngrams
  count(label, bigram, sort = TRUE) %>%
  # Calculate tf-idf
  bind_tf_idf(bigram, label, n) %>%
  arrange(desc(tf_idf)) %>%
  filter(label == 1)

# Select bigrams and clean
bigrams <- bigrams_tf_idf %>%
  # Clean the bigrams
  mutate(
    remove = if_else(str_detect(bigram, regex("\\d|^amp|amp$|tl dr|throwaway account")), 1, 0)
  ) %>%
  filter(remove == 0) %>%
  slice_head(n = 1666) %>%
  select(bigram)

# Save to file
write_csv(bigrams, "data/cleaned/dass/ngrams/bigrams_depression.csv")

#* GET UNIGRAMS -----------------------------------------------------------

# Get unigrams
unigrams_df <- depression_df %>%
  # Find the unigrams
  unnest_ngrams(word, text, n = 1)

# Clean the unigrams
unigrams_tf_idf <- unigrams_df %>%
  # Remove stop words
  filter(!word %in% stop_words$word) %>%
  filter(!(word %in% stopwords(source = "snowball"))) %>%
  filter(!(word %in% stopwords(source = "stopwords-iso"))) %>%
  # Clean up based on remaining stop words
  mutate(
    stop_word = if_else(str_detect(word, regex("^im$|that's|i’m|it’s|you’re|don’t|dont|It|can’t|lt|he’s|she’s|i’ve|doesn’t|didn’t|isn’t|there’s|that'll|how’s|they’ll|it’ll|would've|we’ll|they’ve|shouldn’t|that’s|i’ll|they’re|aren’t|i’d|won’t|what’s|you’ve|we’re|wouldn’t|haven’t|wasn’t|y'all|let’s|here’s|who’s|you’ll|couldn’t|weren’t|hasn’t|we’ve|ain’t|you’d|y’all")), 1, 0) 
  ) %>%
  # Remove remaining stop words
  filter(stop_word == 0) %>%
  select(-stop_word) %>%
  # Get counts of words
  count(label, word, sort = TRUE) %>%
  # Calculate tf-idf
  bind_tf_idf(word, label, n) %>%
  # Get top tf-idf words
  arrange(desc(tf_idf))
unigrams_tf_idf

# Select the positive unigrams
unigrams_pos <- unigrams_tf_idf %>%
  filter(label == 1) %>%
  # Remove unhelpful words
  mutate(remove = if_else(str_detect(word, regex("^\\d*$|\\d\\df|\\d\\dyo|\\dish|\\d'\\d|^\\d*\\.\\d*$")), 1, 0)) %>%
  filter(remove == 0)

# Select the negative unigrams
unigrams_neg <- unigrams_tf_idf %>%
  filter(label == 0) %>%
  # Remove unhelpful words
  mutate(remove = if_else(str_detect(word, regex("^\\d*$|\\d\\df|\\d\\dyo|\\dish|\\d'\\d|^\\d*\\.\\d*$")), 1, 0)) %>%
  filter(remove == 0)

# Remove negative unigrams from positive unigrams
unigrams <- anti_join(unigrams_pos, unigrams_neg) %>%
  # Select the unigrams
  slice_head(n = 1668) %>%
  select(unigram = word) %>%
  bind_rows(data.frame(unigram = c("depressed", "depressing"))) %>%
  distinct(unigram)

# Write to file 
write_csv(unigrams, "data/cleaned/dass/ngrams/unigrams_depression.csv")

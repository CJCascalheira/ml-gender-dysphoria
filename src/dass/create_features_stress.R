# Dependencies
library(tidyverse)
library(tidytext)
library(stopwords)

# Set a random seed
set.seed(1234567)

# Import data
stress_df <- read_csv("data/cleaned/dass/stress_df.csv")

# Load stop words
data(stop_words)

# Number of n-grams per DASS component = 1250
# NOTE had to do fewer in stress because of the poor quality ngrams

# GET FEATURES ------------------------------------------------------------

# Log likelihood ratio test
# http://uc-r.github.io/creating-text-features#likelihood
# https://web.stanford.edu/~jurafsky/slp3/3.pdf
# https://leimao.github.io/blog/Maximum-Likelihood-Estimation-Ngram/
# https://www.youtube.com/watch?v=UyC0bBiZY-A
# https://www.inf.ed.ac.uk/teaching/courses/fnlp/lectures/03_slides.pdf

#* GET TRIGRAMS -----------------------------------------------------------

# Generate and clean trigrams
trigrams_df <- stress_df %>%
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
    stop_word1 = if_else(str_detect(word1, regex("^im$|that's|i’m|it’s|you’re|don’t|dont|It|can’t|lt|he’s|she’s|i’ve|doesn’t|didn’t|isn’t|there’s|that'll|how’s|they’ll|it’ll|would've|we’ll|they’ve|shouldn’t|that’s|i’ll|they’re|aren’t|i’d|won’t|what’s|you’ve|we’re|wouldn’t|haven’t|wasn’t|y'all|let’s|here’s|who’s|you’ll|couldn’t|weren’t|hasn’t|we’ve|ain’t|you’d|y’all|und|ich|dass|die|sie|auch|bek|eine|wenn|binaural|euch|adresse|meine|einen|wmu|selber|jfe|umfrage|^ \\w|amazon|american|amukkara|aoife|odonovan|ucsf|asmr|arghhhhhh|argentinosaurus|ans ")), 1, 0),
    stop_word2 = if_else(str_detect(word2, regex("^im$|that's|i’m|it’s|you’re|don’t|dont|It|can’t|lt|he’s|she’s|i’ve|doesn’t|didn’t|isn’t|there’s|that'll|how’s|they’ll|it’ll|would've|we’ll|they’ve|shouldn’t|that’s|i’ll|they’re|aren’t|i’d|won’t|what’s|you’ve|we’re|wouldn’t|haven’t|wasn’t|y'all|let’s|here’s|who’s|you’ll|couldn’t|weren’t|hasn’t|we’ve|ain’t|you’d|y’all|und|ich|dass|die|sie|auch|bek|eine|wenn|binaural|euch|adresse|meine|einen|wmu|selber|jfe|umfrage|^ \\w|amazon|american|amukkara|aoife|odonovan|ucsf|asmr|arghhhhhh|argentinosaurus|ans ")), 1, 0),
    stop_word3 = if_else(str_detect(word3, regex("^im$|that's|i’m|it’s|you’re|don’t|dont|It|can’t|lt|he’s|she’s|i’ve|doesn’t|didn’t|isn’t|there’s|that'll|how’s|they’ll|it’ll|would've|we’ll|they’ve|shouldn’t|that’s|i’ll|they’re|aren’t|i’d|won’t|what’s|you’ve|we’re|wouldn’t|haven’t|wasn’t|y'all|let’s|here’s|who’s|you’ll|couldn’t|weren’t|hasn’t|we’ve|ain’t|you’d|y’all|und|ich|dass|die|sie|auch|bek|eine|wenn|binaural|euch|adresse|meine|einen|wmu|selber|jfe|umfrage|^ \\w|amazon|american|amukkara|aoife|odonovan|ucsf|asmr|arghhhhhh|argentinosaurus|ans ")), 1, 0)
  ) %>%
  # Remove remaining stop words
  filter(stop_word1 == 0, stop_word2 == 0, stop_word3 == 0) %>%
  select(-stop_word1, -stop_word2, -stop_word3)

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
trigrams <- trigrams_tf_idf %>%
  # Clean the trigrams
  mutate(
    remove = if_else(str_detect(trigram, regex("\\d|^amp|amp$| amp |tl dr|den lille havfrue|dr anju mathur|dr scott gaynor|academically hey guys|afford rent hey|ahsdg dg dfh")), 1, 0)
  ) %>%
  filter(remove == 0) %>%
  slice_head(n = 200) %>%
  select(trigram)

# Save to file
write_csv(trigrams, "data/cleaned/dass/ngrams/trigrams_stress.csv")

#* GET BIGRAMS ------------------------------------------------------------

# Generate and clean bigrams
bigrams_df <- stress_df %>%
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
    stop_word1 = if_else(str_detect(word1, regex("^im$|that's|i’m|it’s|you’re|don’t|dont|It|can’t|lt|he’s|she’s|i’ve|doesn’t|didn’t|isn’t|there’s|that'll|how’s|they’ll|it’ll|would've|we’ll|they’ve|shouldn’t|that’s|i’ll|they’re|aren’t|i’d|won’t|what’s|you’ve|we’re|wouldn’t|haven’t|wasn’t|y'all|let’s|here’s|who’s|you’ll|couldn’t|weren’t|hasn’t|we’ve|ain’t|you’d|y’all|und|ich|dass|die|sie|auch|bek|eine|wenn|binaural|euch|adresse|meine|einen|wmu|selber|jfe|umfrage|^ \\w|amazon|american|amukkara|aoife|odonovan|ucsf|asmr|arghhhhhh|argentinosaurus|ans |survey|gift")), 1, 0),
    stop_word2 = if_else(str_detect(word2, regex("^im$|that's|i’m|it’s|you’re|don’t|dont|It|can’t|lt|he’s|she’s|i’ve|doesn’t|didn’t|isn’t|there’s|that'll|how’s|they’ll|it’ll|would've|we’ll|they’ve|shouldn’t|that’s|i’ll|they’re|aren’t|i’d|won’t|what’s|you’ve|we’re|wouldn’t|haven’t|wasn’t|y'all|let’s|here’s|who’s|you’ll|couldn’t|weren’t|hasn’t|we’ve|ain’t|you’d|y’all|und|ich|dass|die|sie|auch|bek|eine|wenn|binaural|euch|adresse|meine|einen|wmu|selber|jfe|umfrage|^ \\w|amazon|american|amukkara|aoife|odonovan|ucsf|asmr|arghhhhhh|argentinosaurus|ans |survey|gift")), 1, 0)
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
    remove = if_else(str_detect(bigram, regex("\\d|^amp|amp$|tl dr|throwaway account|dp dr|irc channel|irc$|survey items|gift card|totally fine")), 1, 0)
  ) %>%
  filter(remove == 0) %>%
  slice_head(n = 200) %>%
  select(bigram)

# Save to file
write_csv(bigrams, "data/cleaned/dass/ngrams/bigrams_stress.csv")

#* GET UNIGRAMS -----------------------------------------------------------

# Get unigrams
unigrams_df <- stress_df %>%
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
    stop_word = if_else(str_detect(word, regex("^im$|that's|i’m|it’s|you’re|don’t|dont|It|can’t|lt|he’s|she’s|i’ve|doesn’t|didn’t|isn’t|there’s|that'll|how’s|they’ll|it’ll|would've|we’ll|they’ve|shouldn’t|that’s|i’ll|they’re|aren’t|i’d|won’t|what’s|you’ve|we’re|wouldn’t|haven’t|wasn’t|y'all|let’s|here’s|who’s|you’ll|couldn’t|weren’t|hasn’t|we’ve|ain’t|you’d|y’all|und|ich|dass|die|sie|auch|bek|eine|wenn|binaural|euch|adresse|meine|einen|wmu|selber|jfe|umfrage|^ \\w|amazon|american|amukkara|aoife|odonovan|ucsf|asmr|arghhhhhh|argentinosaurus|ans |survey|gift")), 1, 0) 
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
  mutate(remove = if_else(str_detect(word, regex("^\\d*$|\\d\\df|\\d\\dyo|\\dish|\\d'\\d|^\\d*\\.\\d*$|skype|forms|notes|ap|apps|bc|idk|monday|tuesday|wednesday|thursday|friday|saturday|sunday|shut|greatly|uni$")), 1, 0)) %>%
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
  slice_head(n = 199) %>%
  select(unigram = word) %>%
  bind_rows(data.frame(unigram = c("stressed", "stressful"))) %>%
  distinct(unigram)

# Write to file 
write_csv(unigrams, "data/cleaned/dass/ngrams/unigrams_stress.csv")

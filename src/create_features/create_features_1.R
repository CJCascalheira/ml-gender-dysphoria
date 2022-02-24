# Dependencies
library(tidyverse)
library(tidytext)

# Import data
df_train <- read_csv("data/cleaned/df_train_clean.csv")
df_test <- read_csv("data/cleaned/df_test_clean.csv")
dsm5 <- read_csv("data/dsm5_gender_dysphoria.csv")

# Load stop words
data("stop_words")
stop_words

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
df_train <- df_train %>%
  mutate(
    clinical_keywords = if_else(str_detect(text, regex("dysphor*|natal|disorder|assign*|develop*")), 1, 0)
  )

# Add features - test
df_test <- df_test %>%
  mutate(
    clinical_keywords = if_else(str_detect(text, regex("dysphor*|natal|disorder|assign*|develop*")), 1, 0)
  )

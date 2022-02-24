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

# SAVE DATAFRAMES WITH FEATURES -------------------------------------------

write_csv(df_train, "data/cleaned/with_features/df_train.csv")
write_csv(df_test, "data/cleaned/with_features/df_test.csv")

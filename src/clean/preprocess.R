# Dependencies
library(tidyverse)

# Import
df_train_raw <- read_csv("data/raw/df_train_raw.csv")
df_test_raw <- read_csv("data/raw/df_test_raw.csv")

# CLEAN TRAINING DATA -----------------------------------------------------

# Clean up test data
df_train <- df_train_raw %>%
  filter(!is.na(text)) %>%
  filter(text != "[deleted]", text != "[removed]") %>%
  # Remove links / URLs
  mutate(text = str_remove_all(text, " ?(f|ht)tp(s?)://(.*)[.][a-z]+")) %>%
  # Replace whitespace characters
  mutate(text = str_replace_all(text, "\r\n\r\n", " ")) %>%
  # Remove strange characters
  mutate(text = str_remove_all(text, "&amp;#x200B;|â€¦|&lt;|&gt;|â€œ|ðŸ¥´|ðŸ¥²|â„¢|ðŸ¤·â€|â™€ï¸|â€™|â€|&gt;|Ã©||ðŸ™|ðŸŒˆ|ðŸ")) %>%
  # Recode characters
  mutate(text = recode(text, "&amp;" = "and", "Â´" = "'", "â€™" = "'")) %>%
  # Lowercase
  mutate(text = str_to_lower(text))
df_train

# Check for empty strings
df_train %>%
  mutate(str_len = str_length(text)) %>%
  arrange(str_len)

# CLEAN TESTING DATA ------------------------------------------------------

# Clean up text data
df_test <- df_test_raw %>%
  filter(!is.na(text)) %>%
  filter(text != "[deleted]", text != "[removed]") %>%
  # Remove links / URLs
  mutate(text = str_remove_all(text, " ?(f|ht)tp(s?)://(.*)[.][a-z]+")) %>%
  # Replace whitespace characters
  mutate(text = str_replace_all(text, "\r\n\r\n", " ")) %>%
  # Remove strange characters
  mutate(text = str_remove_all(text, "&amp;#x200B;|â€¦|&lt;|&gt;|â€œ|ðŸ¥´|ðŸ¥²|â„¢|ðŸ¤·â€|â™€ï¸|â€™|â€|&gt;|Ã©||ðŸ™|ðŸŒˆ|ðŸ")) %>%
  # Recode characters
  mutate(text = recode(text, "&amp;" = "and", "Â´" = "'", "â€™" = "'")) %>%
  # Lowercase
  mutate(text = str_to_lower(text))
df_test

# Check for empty strings
df_test %>%
  mutate(str_len = str_length(text)) %>%
  arrange(str_len)

# Clean up part 2
df_test_1 <- df_test %>%
  mutate(text = str_trim(text)) %>%
  # Remove empty strings
  filter(text != " ", text != "")

# Check for empty strings
df_test_1 %>%
  mutate(str_len = str_length(text)) %>%
  arrange(str_len)

# WRITE TO FILE -----------------------------------------------------------

write_csv(df_train, "data/cleaned/df_train_clean.csv")
write_csv(df_test_1, "data/cleaned/df_test_clean.csv")

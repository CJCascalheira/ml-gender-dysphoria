# Dependencies
library(tidyverse)

# Import gender dysphoria subreddit
df_dysphoria <- read_csv("data/raw/reddit_dysphoria/reddit_dysphoria.csv")

# Set seed
set.seed(1234567)

# Initial clean
df_truth_dysphoria <- df_dysphoria %>%
  # Select columns
  select(text, title) %>%
  # Remove reddit-specific nonsense
  filter(!text %in% c("[deleted]", "[removed]")) %>%
  # Unite the title and text columns
  unite(text, c(title, text), sep = " ") %>%
  # Remove missing
  filter(!is.na(text)) %>%
  # Add positive label for all posts in Gender Dysphoria subreddit
  mutate(dysphoria = rep("NA", nrow(.))) %>%
  mutate(
    # Remove unhelpful labels
    remove = if_else(str_detect(text, regex("selling|research|need mods|looking for mods|discord server|pm.*discord",
                                            ignore_case = TRUE)), 1, 0),
    # Remove posts with fewer than 100 characters
    str_len = str_length(text)
  ) %>%
  filter(remove == 0, str_len > 100) %>%
  select(-remove, -str_len) %>%
  # Determine who is coding the posts
  mutate(who = rep(c("alejandra", "andre", "emily", "jasmine", "dannie", "danica"), length.out = nrow(.))) %>%
  # Select the columns for merging
  select(text, who, dysphoria)

# How many posts for each coder?
df_truth_dysphoria %>%
  count(who)

# Export for coder verification
write_csv(df_truth_dysphoria, "data/cleaned/dysphoria_reddit/df_truth_dysphoria_uncoded.csv")

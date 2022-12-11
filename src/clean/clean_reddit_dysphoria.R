# Dependencies
library(tidyverse)
library(rio)

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

# MERGE CLEANED -----------------------------------------------------------

# Import and clean up data for each coder
emily <- read_csv("data/cleaned/dysphoria_reddit/df_truth_dysphoria_uncoded.csv") %>%
  filter(who == "emily") %>%
  mutate(dysphoria = if_else(is.na(dysphoria), 1, 0))

jasmine <- import("data/cleaned/dysphoria_reddit/jasmine.xlsx") %>% 
  as_tibble() %>%
  filter(who == "jasmine") %>%
  mutate(dysphoria = if_else(str_detect(dysphoria, regex("yes|na", ignore_case = TRUE)), 1, 0))
jasmine <- jasmine[, 1:3]

dannie <- import("data/cleaned/dysphoria_reddit/dannie.xlsx") %>% 
  as_tibble() %>%
  filter(who == "dannie") %>%
  mutate(dysphoria = if_else(dysphoria == "1", 1, 0))
dannie <- dannie[, 1:3]

danica <- import("data/cleaned/dysphoria_reddit/danica.xlsx") %>% as_tibble()
danica <- danica[-1, 1:3]
names(danica) <- c("text", "who", "dysphoria")
danica <- danica %>% 
  filter(who == "danica") %>%
  mutate(dysphoria = if_else(dysphoria == "1", 1, 0))

andre <- read_csv("data/cleaned/dysphoria_reddit/andre.csv") %>%
  filter(who == "andre")

alejandra <- import("data/cleaned/dysphoria_reddit/alejandra.xlsx") %>% 
  as_tibble() %>%
  filter(who == "alejandra") %>%
  mutate(dysphoria = if_else(dysphoria == "1", 1, 0))

# Bind coded data
df_dysphoria_reddit <- bind_rows(emily, jasmine) %>%
  bind_rows(dannie) %>%
  bind_rows(danica) %>%
  bind_rows(andre) %>%
  bind_rows(alejandra) %>%
  select(-who)

# Write to file
write_csv(df_dysphoria_reddit, "data/cleaned/dysphoria_reddit/df_truth_dysphoria_coded.csv")

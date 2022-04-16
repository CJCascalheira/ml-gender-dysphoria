# Dependencies
library(tidyverse)

# Import DASS datasets
depression_df <- read_csv("data/cleaned/dass/depression_df.csv")
anxiety_df <- read_csv("data/cleaned/dass/anxiety_df.csv")
stress_df <- read_csv("data/cleaned/dass/stress_df.csv")
suicide_df <- read_csv("data/cleaned/dass/suicide_df.csv")

# Import unigrams
my_csvs <- list.files("data/cleaned/dass/ngrams/")
my_files1 <- paste0("data/cleaned/dass/ngrams/", my_csvs)
files_list <- lapply(my_files1, read_csv)

# Add DASS names 
for (i in 1:length(my_files1)) {
  files_list[[i]] <- files_list[[i]] %>%
    mutate(dass = rep(my_files1[i], nrow(.)))
}

# Select unigrams
unigram_df <- files_list[9:12] %>%
  # Merge into one dataframe
  bind_rows() %>%
  # Clean the DASS names
  mutate(dass = str_extract(dass, regex("(?<=data/cleaned/dass/ngrams/unigrams_)\\w*(?=.csv)")))

# Select bigrams
bigram_df <- files_list[1:4] %>%
  # Merge into one dataframe
  bind_rows() %>%
  # Clean the DASS names
  mutate(dass = str_extract(dass, regex("(?<=data/cleaned/dass/ngrams/bigrams_)\\w*(?=.csv)")))

# Select trigrams
trigram_df <- files_list[5:8] %>%
  # Merge into one dataframe
  bind_rows() %>%
  # Clean the DASS names
  mutate(dass = str_extract(dass, regex("(?<=data/cleaned/dass/ngrams/trigrams_)\\w*(?=.csv)")))

# ADD FEATURES TO EACH DASS DATAFRAME -------------------------------------

#* DEPRESSION -------------------------------------------------------------

# Get just the unigrams
unigram_depression <- unigram_df %>%
  filter(dass == "depression") %>%
  pull(unigram)

# Get just the bigrams
bigram_depression <- bigram_df %>%
  filter(dass == "depression") %>%
  pull(bigram)

# Get just the trigrams
trigram_depression <- trigram_df %>%
  filter(dass == "depression") %>%
  pull(trigram)

# Assign the unigrams as features
for (i in 1:length(unigram_depression)) {
  
  # Get the n-grams
  ngram <- unigram_depression[i]
  
  # Detect the n-gram with regular expressions
  x <- str_detect(depression_df$text, regex(ngram))
  
  # Add the n-gram to the dataframe
  depression_df[[ngram]] <- as.integer(x)  
}

# Assign the bigrams as features
for (i in 1:length(bigram_depression)) {
  
  # Get the n-grams
  ngram <- bigram_depression[i]
  
  # Detect the n-gram with regular expressions
  x <- str_detect(depression_df$text, regex(ngram))
  
  # Add the n-gram to the dataframe
  depression_df[[ngram]] <- as.integer(x)  
}

# Assign the trigrams as features
for (i in 1:length(trigram_depression)) {
  
  # Get the n-grams
  ngram <- trigram_depression[i]
  
  # Detect the n-gram with regular expressions
  x <- str_detect(depression_df$text, regex(ngram))
  
  # Add the n-gram to the dataframe
  depression_df[[ngram]] <- as.integer(x)  
}

# Save to file
write_csv(depression_df, "data/cleaned/dass/with_features/depression.csv")

#* ANXIETY ----------------------------------------------------------------

# Get just the unigrams
unigram_anxiety <- unigram_df %>%
  filter(dass == "anxiety") %>%
  pull(unigram)

# Get just the bigrams
bigram_anxiety <- bigram_df %>%
  filter(dass == "anxiety") %>%
  pull(bigram)

# Get just the trigrams
trigram_anxiety <- trigram_df %>%
  filter(dass == "anxiety") %>%
  pull(trigram)

# Assign the unigrams as features
for (i in 1:length(unigram_anxiety)) {
  
  # Get the n-grams
  ngram <- unigram_anxiety[i]
  
  # Detect the n-gram with regular expressions
  x <- str_detect(anxiety_df$text, regex(ngram))
  
  # Add the n-gram to the dataframe
  anxiety_df[[ngram]] <- as.integer(x)  
}

# Assign the bigrams as features
for (i in 1:length(bigram_anxiety)) {
  
  # Get the n-grams
  ngram <- bigram_anxiety[i]
  
  # Detect the n-gram with regular expressions
  x <- str_detect(anxiety_df$text, regex(ngram))
  
  # Add the n-gram to the dataframe
  anxiety_df[[ngram]] <- as.integer(x)  
}

# Assign the trigrams as features
for (i in 1:length(trigram_anxiety)) {
  
  # Get the n-grams
  ngram <- trigram_anxiety[i]
  
  # Detect the n-gram with regular expressions
  x <- str_detect(anxiety_df$text, regex(ngram))
  
  # Add the n-gram to the dataframe
  anxiety_df[[ngram]] <- as.integer(x)  
}

# Save to file
write_csv(anxiety_df, "data/cleaned/dass/with_features/anxiety.csv")

#* STRESS -----------------------------------------------------------------

# Get just the unigrams
unigram_stress <- unigram_df %>%
  filter(dass == "stress") %>%
  pull(unigram)

# Get just the bigrams
bigram_stress <- bigram_df %>%
  filter(dass == "stress") %>%
  pull(bigram)

# Get just the trigrams
trigram_stress <- trigram_df %>%
  filter(dass == "stress") %>%
  pull(trigram)

# Assign the unigrams as features
for (i in 1:length(unigram_stress)) {
  
  # Get the n-grams
  ngram <- unigram_stress[i]
  
  # Detect the n-gram with regular expressions
  x <- str_detect(stress_df$text, regex(ngram))
  
  # Add the n-gram to the dataframe
  stress_df[[ngram]] <- as.integer(x)  
}

# Assign the bigrams as features
for (i in 1:length(bigram_stress)) {
  
  # Get the n-grams
  ngram <- bigram_stress[i]
  
  # Detect the n-gram with regular expressions
  x <- str_detect(stress_df$text, regex(ngram))
  
  # Add the n-gram to the dataframe
  stress_df[[ngram]] <- as.integer(x)  
}

# Assign the trigrams as features
for (i in 1:length(trigram_stress)) {
  
  # Get the n-grams
  ngram <- trigram_stress[i]
  
  # Detect the n-gram with regular expressions
  x <- str_detect(stress_df$text, regex(ngram))
  
  # Add the n-gram to the dataframe
  stress_df[[ngram]] <- as.integer(x)  
}

# Save to file
write_csv(stress_df, "data/cleaned/dass/with_features/stress.csv")

#* SUICIDE ----------------------------------------------------------------

# Get just the unigrams
unigram_suicide <- unigram_df %>%
  filter(dass == "suicide") %>%
  pull(unigram)

# Get just the bigrams
bigram_suicide <- bigram_df %>%
  filter(dass == "suicide") %>%
  pull(bigram)

# Get just the trigrams
trigram_suicide <- trigram_df %>%
  filter(dass == "suicide") %>%
  pull(trigram)

# Assign the unigrams as features
for (i in 1:length(unigram_suicide)) {
  
  # Get the n-grams
  ngram <- unigram_suicide[i]
  
  # Detect the n-gram with regular expressions
  x <- str_detect(suicide_df$text, regex(ngram))
  
  # Add the n-gram to the dataframe
  suicide_df[[ngram]] <- as.integer(x)  
}

# Assign the bigrams as features
for (i in 1:length(bigram_suicide)) {
  
  # Get the n-grams
  ngram <- bigram_suicide[i]
  
  # Detect the n-gram with regular expressions
  x <- str_detect(suicide_df$text, regex(ngram))
  
  # Add the n-gram to the dataframe
  suicide_df[[ngram]] <- as.integer(x)  
}

# Assign the trigrams as features
for (i in 1:length(trigram_suicide)) {
  
  # Get the n-grams
  ngram <- trigram_suicide[i]
  
  # Detect the n-gram with regular expressions
  x <- str_detect(suicide_df$text, regex(ngram))
  
  # Add the n-gram to the dataframe
  suicide_df[[ngram]] <- as.integer(x)  
}

# Save to file
write_csv(suicide_df, "data/cleaned/dass/with_features/suicide.csv")

# ADD FEATURES TO GROUND TRUTH AND PRIMARY --------------------------------

#* DEPRESSION -------------------------------------------------------------

# Import ground truth and primary datasets
df_truth <- read_csv("data/cleaned/df_truth_clean.csv")
# df_primary <- read_csv("data/cleaned/df_primary_clean.csv")

# NOTE: Comment out the df_primary because the set.seed may not have worked in 
# the preprocess.R file. We already assigned DASS features to the df_primary 
# dataset; just re-assigning to df_truth because we updated the dataset.

# GROUND TRUTH DATASET

# Assign the unigrams as features
for (i in 1:length(unigram_depression)) {
  
  # Get the n-grams
  ngram <- unigram_depression[i]
  
  # Detect the n-gram with regular expressions
  x <- str_detect(df_truth$text, regex(ngram))
  
  # Add the n-gram to the dataframe
  df_truth[[ngram]] <- as.integer(x)  
}

# Assign the bigrams as features
for (i in 1:length(bigram_depression)) {
  
  # Get the n-grams
  ngram <- bigram_depression[i]
  
  # Detect the n-gram with regular expressions
  x <- str_detect(df_truth$text, regex(ngram))
  
  # Add the n-gram to the dataframe
  df_truth[[ngram]] <- as.integer(x)  
}

# Assign the trigrams as features
for (i in 1:length(trigram_depression)) {
  
  # Get the n-grams
  ngram <- trigram_depression[i]
  
  # Detect the n-gram with regular expressions
  x <- str_detect(df_truth$text, regex(ngram))
  
  # Add the n-gram to the dataframe
  df_truth[[ngram]] <- as.integer(x)  
}

# Save to file
write_csv(df_truth, "data/cleaned/dass/truth/truth_depression.csv")

# PRIMARY DATASET

# Assign the unigrams as features
for (i in 1:length(unigram_depression)) {
  
  # Get the n-grams
  ngram <- unigram_depression[i]
  
  # Detect the n-gram with regular expressions
  x <- str_detect(df_primary$text, regex(ngram))
  
  # Add the n-gram to the dataframe
  df_primary[[ngram]] <- as.integer(x)  
}

# Assign the bigrams as features
for (i in 1:length(bigram_depression)) {
  
  # Get the n-grams
  ngram <- bigram_depression[i]
  
  # Detect the n-gram with regular expressions
  x <- str_detect(df_primary$text, regex(ngram))
  
  # Add the n-gram to the dataframe
  df_primary[[ngram]] <- as.integer(x)  
}

# Assign the trigrams as features
for (i in 1:length(trigram_depression)) {
  
  # Get the n-grams
  ngram <- trigram_depression[i]
  
  # Detect the n-gram with regular expressions
  x <- str_detect(df_primary$text, regex(ngram))
  
  # Add the n-gram to the dataframe
  df_primary[[ngram]] <- as.integer(x)  
}

# Save to file
write_csv(df_primary, "data/cleaned/dass/primary/primary_depression.csv")

#* ANXIETY ----------------------------------------------------------------

# Import ground truth and primary datasets
df_truth <- read_csv("data/cleaned/df_truth_clean.csv")
# df_primary <- read_csv("data/cleaned/df_primary_clean.csv")

# NOTE: Comment out the df_primary because the set.seed may not have worked in 
# the preprocess.R file. We already assigned DASS features to the df_primary 
# dataset; just re-assigning to df_truth because we updated the dataset.

# GROUND TRUTH DATASET

# Assign the unigrams as features
for (i in 1:length(unigram_anxiety)) {
  
  # Get the n-grams
  ngram <- unigram_anxiety[i]
  
  # Detect the n-gram with regular expressions
  x <- str_detect(df_truth$text, regex(ngram))
  
  # Add the n-gram to the dataframe
  df_truth[[ngram]] <- as.integer(x)  
}

# Assign the bigrams as features
for (i in 1:length(bigram_anxiety)) {
  
  # Get the n-grams
  ngram <- bigram_anxiety[i]
  
  # Detect the n-gram with regular expressions
  x <- str_detect(df_truth$text, regex(ngram))
  
  # Add the n-gram to the dataframe
  df_truth[[ngram]] <- as.integer(x)  
}

# Assign the trigrams as features
for (i in 1:length(trigram_anxiety)) {
  
  # Get the n-grams
  ngram <- trigram_anxiety[i]
  
  # Detect the n-gram with regular expressions
  x <- str_detect(df_truth$text, regex(ngram))
  
  # Add the n-gram to the dataframe
  df_truth[[ngram]] <- as.integer(x)  
}

# Save to file
write_csv(df_truth, "data/cleaned/dass/truth/truth_anxiety.csv")

# PRIMARY DATASET

# Assign the unigrams as features
for (i in 1:length(unigram_anxiety)) {
  
  # Get the n-grams
  ngram <- unigram_anxiety[i]
  
  # Detect the n-gram with regular expressions
  x <- str_detect(df_primary$text, regex(ngram))
  
  # Add the n-gram to the dataframe
  df_primary[[ngram]] <- as.integer(x)  
}

# Assign the bigrams as features
for (i in 1:length(bigram_anxiety)) {
  
  # Get the n-grams
  ngram <- bigram_anxiety[i]
  
  # Detect the n-gram with regular expressions
  x <- str_detect(df_primary$text, regex(ngram))
  
  # Add the n-gram to the dataframe
  df_primary[[ngram]] <- as.integer(x)  
}

# Assign the trigrams as features
for (i in 1:length(trigram_anxiety)) {
  
  # Get the n-grams
  ngram <- trigram_anxiety[i]
  
  # Detect the n-gram with regular expressions
  x <- str_detect(df_primary$text, regex(ngram))
  
  # Add the n-gram to the dataframe
  df_primary[[ngram]] <- as.integer(x)  
}

# Save to file
write_csv(df_primary, "data/cleaned/dass/primary/primary_anxiety.csv")

#* STRESS -----------------------------------------------------------------

# Import ground truth and primary datasets
df_truth <- read_csv("data/cleaned/df_truth_clean.csv")
# df_primary <- read_csv("data/cleaned/df_primary_clean.csv")

# NOTE: Comment out the df_primary because the set.seed may not have worked in 
# the preprocess.R file. We already assigned DASS features to the df_primary 
# dataset; just re-assigning to df_truth because we updated the dataset.

# GROUND TRUTH DATASET

# Assign the unigrams as features
for (i in 1:length(unigram_stress)) {
  
  # Get the n-grams
  ngram <- unigram_stress[i]
  
  # Detect the n-gram with regular expressions
  x <- str_detect(df_truth$text, regex(ngram))
  
  # Add the n-gram to the dataframe
  df_truth[[ngram]] <- as.integer(x)  
}

# Assign the bigrams as features
for (i in 1:length(bigram_stress)) {
  
  # Get the n-grams
  ngram <- bigram_stress[i]
  
  # Detect the n-gram with regular expressions
  x <- str_detect(df_truth$text, regex(ngram))
  
  # Add the n-gram to the dataframe
  df_truth[[ngram]] <- as.integer(x)  
}

# Assign the trigrams as features
for (i in 1:length(trigram_stress)) {
  
  # Get the n-grams
  ngram <- trigram_stress[i]
  
  # Detect the n-gram with regular expressions
  x <- str_detect(df_truth$text, regex(ngram))
  
  # Add the n-gram to the dataframe
  df_truth[[ngram]] <- as.integer(x)  
}

# Save to file
write_csv(df_truth, "data/cleaned/dass/truth/truth_stress.csv")

# PRIMARY DATASET

# Assign the unigrams as features
for (i in 1:length(unigram_stress)) {
  
  # Get the n-grams
  ngram <- unigram_stress[i]
  
  # Detect the n-gram with regular expressions
  x <- str_detect(df_primary$text, regex(ngram))
  
  # Add the n-gram to the dataframe
  df_primary[[ngram]] <- as.integer(x)  
}

# Assign the bigrams as features
for (i in 1:length(bigram_stress)) {
  
  # Get the n-grams
  ngram <- bigram_stress[i]
  
  # Detect the n-gram with regular expressions
  x <- str_detect(df_primary$text, regex(ngram))
  
  # Add the n-gram to the dataframe
  df_primary[[ngram]] <- as.integer(x)  
}

# Assign the trigrams as features
for (i in 1:length(trigram_stress)) {
  
  # Get the n-grams
  ngram <- trigram_stress[i]
  
  # Detect the n-gram with regular expressions
  x <- str_detect(df_primary$text, regex(ngram))
  
  # Add the n-gram to the dataframe
  df_primary[[ngram]] <- as.integer(x)  
}

# Save to file
write_csv(df_primary, "data/cleaned/dass/primary/primary_stress.csv")

#* SUICIDE ----------------------------------------------------------------

# Import ground truth and primary datasets
df_truth <- read_csv("data/cleaned/df_truth_clean.csv")
# df_primary <- read_csv("data/cleaned/df_primary_clean.csv")

# NOTE: Comment out the df_primary because the set.seed may not have worked in 
# the preprocess.R file. We already assigned DASS features to the df_primary 
# dataset; just re-assigning to df_truth because we updated the dataset.

# GROUND TRUTH DATASET

# Assign the unigrams as features
for (i in 1:length(unigram_suicide)) {
  
  # Get the n-grams
  ngram <- unigram_suicide[i]
  
  # Detect the n-gram with regular expressions
  x <- str_detect(df_truth$text, regex(ngram))
  
  # Add the n-gram to the dataframe
  df_truth[[ngram]] <- as.integer(x)  
}

# Assign the bigrams as features
for (i in 1:length(bigram_suicide)) {
  
  # Get the n-grams
  ngram <- bigram_suicide[i]
  
  # Detect the n-gram with regular expressions
  x <- str_detect(df_truth$text, regex(ngram))
  
  # Add the n-gram to the dataframe
  df_truth[[ngram]] <- as.integer(x)  
}

# Assign the trigrams as features
for (i in 1:length(trigram_suicide)) {
  
  # Get the n-grams
  ngram <- trigram_suicide[i]
  
  # Detect the n-gram with regular expressions
  x <- str_detect(df_truth$text, regex(ngram))
  
  # Add the n-gram to the dataframe
  df_truth[[ngram]] <- as.integer(x)  
}

# Save to file
write_csv(df_truth, "data/cleaned/dass/truth/truth_suicide.csv")

# PRIMARY DATASET

# Assign the unigrams as features
for (i in 1:length(unigram_suicide)) {
  
  # Get the n-grams
  ngram <- unigram_suicide[i]
  
  # Detect the n-gram with regular expressions
  x <- str_detect(df_primary$text, regex(ngram))
  
  # Add the n-gram to the dataframe
  df_primary[[ngram]] <- as.integer(x)  
}

# Assign the bigrams as features
for (i in 1:length(bigram_suicide)) {
  
  # Get the n-grams
  ngram <- bigram_suicide[i]
  
  # Detect the n-gram with regular expressions
  x <- str_detect(df_primary$text, regex(ngram))
  
  # Add the n-gram to the dataframe
  df_primary[[ngram]] <- as.integer(x)  
}

# Assign the trigrams as features
for (i in 1:length(trigram_suicide)) {
  
  # Get the n-grams
  ngram <- trigram_suicide[i]
  
  # Detect the n-gram with regular expressions
  x <- str_detect(df_primary$text, regex(ngram))
  
  # Add the n-gram to the dataframe
  df_primary[[ngram]] <- as.integer(x)  
}

# Save to file
write_csv(df_primary, "data/cleaned/dass/primary/primary_suicide.csv")

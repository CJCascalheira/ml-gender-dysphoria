# Dependencies
library(tidyverse)

# Import data
df_truth <- read_csv("data/cleaned/features_temp/df_truth.csv")

# Import data - truth 
depression_truth <- read_csv("data/cleaned/dass/truth/features/y_depression.csv") %>%
  select(dass_depression = `0`)

anxiety_truth <- read_csv("data/cleaned/dass/truth/features/y_anxiety.csv") %>%
  select(dass_anxiety = `0`)

stress_truth <- read_csv("data/cleaned/dass/truth/features/y_stress.csv") %>%
  select(dass_stress = `0`)

suicide_truth <- read_csv("data/cleaned/dass/truth/features/y_suicide.csv") %>%
  select(dass_suicide = `0`)

# Bind columns for truth dataset
df_truth_dass <- bind_cols(df_truth, depression_truth) %>%
  bind_cols(anxiety_truth) %>%
  bind_cols(stress_truth) %>%
  bind_cols(suicide_truth)

# Save file
write_csv(df_truth_dass, "data/cleaned/features_temp/df_truth_dass.csv")

# PRIMARY DATASET ---------------------------------------------------------

# Import data
df_primary <- read_csv("data/cleaned/features_temp/df_primary.csv")

# Import DASS data
depression_primary <- read_csv("data/cleaned/dass/primary/features/y_depression.csv") %>%
  select(dass_depression = `0`)
depression_full <- read_csv("data/cleaned/dass/primary/primary_depression.csv") %>%
  select(text)

# Bind columns
depression_bound <- bind_cols(depression_full,depression_primary)
df_primary2 <- left_join(df_primary, depression_bound, by = "text")

# Clear up space
rm(df_primary, depression_bound, depression_full, depression_primary)

# Import DASS data
anxiety_primary <- read_csv("data/cleaned/dass/primary/features/y_anxiety.csv") %>%
  select(dass_anxiety = `0`)
anxiety_full <- read_csv("data/cleaned/dass/primary/primary_anxiety.csv") %>%
  select(text)

# Bind columns
anxiety_bound <- bind_cols(anxiety_full, anxiety_primary) 
df_primary3 <- left_join(df_primary2, anxiety_bound, by = "text")

# Clear up space
rm(df_primary2, anxiety_bound, anxiety_full, anxiety_primary)

# Import DASS data
stress_primary <- read_csv("data/cleaned/dass/primary/features/y_stress.csv") %>%
  select(dass_stress = `0`)
stress_full <- read_csv("data/cleaned/dass/primary/primary_stress.csv") %>%
  select(text)

# Bind columns
stress_bound <- bind_cols(stress_full, stress_primary)
df_primary4 <- left_join(df_primary3, stress_bound, by = "text")

# Clear up space
rm(df_primary3, stress_bound, stress_full, stress_primary)

# Import DASS data
suicide_primary <- read_csv("data/cleaned/dass/primary/features/y_suicide.csv") %>%
  select(dass_suicide = `0`)
suicide_full <- read_csv("data/cleaned/dass/primary/primary_suicide.csv") %>%
  select(text)

# Bind columns 
suicide_bound <- bind_cols(suicide_full, suicide_primary)
df_primary_dass <- left_join(df_primary4, suicide_bound, by = "text")

# Check work
names(df_primary_dass)

# Save to file
write_csv(df_primary_dass, "data/cleaned/features_temp/df_primary_dass.csv")

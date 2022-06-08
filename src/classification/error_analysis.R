# Load dependencies
library(tidyverse)
library(readxl)

# Import content analysis 
error_df <- read_excel("data/results/xgb_misclassification_content_analysis.xlsx", 
                       sheet = 1)

category_df <- read_excel("data/results/xgb_misclassification_content_analysis.xlsx", 
                          sheet = 2) %>%
  select(category, meta_category)

# Count the categories by error type
category_by_error_type <- error_df %>%
  # Pivot categories to one column
  pivot_longer(cols = coder_error:sex_body, names_to = "category", values_to = "category_value") %>%
  # Join with the other data frame
  left_join(category_df, by = "category") %>%
  # Pivot errors to one column
  pivot_longer(cols = false_pos:false_neg, names_to = "error_type", values_to = "error_value") %>%
  # Keep presence of errors
  filter(error_value == 1) %>%
  # Group the categories by error type
  group_by(error_type) %>%
  filter(category_value == 1) %>%
  # Calculate summary metrics
  count(meta_category) %>%
  mutate(
    total = sum(n),
    percent = round((n / total) * 100, 3)
  ) %>%
  arrange(error_type, desc(percent))
category_by_error_type  

# Get raw categories
total_categories <- error_df %>%
  # Pivot categories to one column
  pivot_longer(cols = coder_error:sex_body, names_to = "category", values_to = "category_value") %>%
  # Join with the other data frame
  left_join(category_df, by = "category") %>%
  # Pivot errors to one column
  pivot_longer(cols = false_pos:false_neg, names_to = "error_type", values_to = "error_value") %>%
  # Apply filters
  filter(error_value == 1) %>%
  filter(category_value == 1) %>%
  # Calculate summary metrics
  count(meta_category) %>%
  mutate(
    total = sum(n),
    percent = round((n / total) * 100, 3)
  ) %>%
  arrange(desc(percent))
total_categories

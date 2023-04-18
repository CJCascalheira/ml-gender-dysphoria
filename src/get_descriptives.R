# Load dependencies
library(tidyverse)

# Import ground truth
df_truth <- read_csv("data/cleaned/with_features/df_truth.csv")

# Word count / length of posts
df_truth %>%
  summarize(
    m_words = mean(WC),
    med_words = median(WC),
    min_words = min(WC),
    max_words = max(WC)
  )

# Visualize
ggplot(df_truth, aes(x = WC)) +
  geom_boxplot(fill = "royalblue2") + 
  theme_bw() +
  labs(x = "Word Count")

# Load dependencies
library(tidyverse)

# Import ground truth
df_truth <- read_csv("data/cleaned/with_features/df_truth.csv")

# Import data and arrange
feat_selected <- read_csv("data/results/feature_selection/number_features_selected.csv") %>%
  arrange(desc(percent_used)) %>%
  rename(category = index)

feat_importance <- read_csv("data/results/feature_selection/avg_feature_importance.csv") %>%
  arrange(desc(info_gained))

# EXAMPLES FOR DL PAPER ---------------------------------------------------

# Get the examples
truth_examples <- df_truth %>%
  group_by(dysphoria) %>%
  select(temp_id, text) %>%
  sample_n(size = 2)

# Save examples
write_csv(truth_examples, "data/results/ground_truth_examples_2.csv")

# FEATURE IMPORTANCE ------------------------------------------------------

# Visualize the number of features selected
feat_selected_plot <- feat_selected %>%
  mutate(category = recode(category, "clinical_keywords" = "Clinical Keywords", "n_grams" = "n-Grams", 
                           "psycholinguistic" = "Psycholinguistic\nAttributes", "sentiment_valence" = "Emotional Valence\n(Sentiment)",
                           "embedding" = "Word Emeddings", "dass" = "Psychological\nDistress (DASS)")) %>%
  mutate(percent_used = round(percent_used, 2)) %>%
  ggplot(aes(x = reorder(category, percent_used), y = percent_used)) +
  geom_bar(stat="identity", fill = "dodgerblue3") +
  coord_flip(clip = "off") +
  scale_y_continuous(expand = c(0, 0)) +
  labs(
    x = "Independent Variable Categories",
    y = "Percent Used in XGBoost Classifier"
  ) +
  theme_bw() +
  theme(text=element_text(family="serif"))
feat_selected_plot

# Visualize the average information gained
feat_importance_plot <- feat_importance %>%
  mutate(category = recode(category, "clinical_keywords" = "Clinical Keywords", "n_grams" = "n-Grams", 
                           "psycholinguistic" = "Psycholinguistic\nAttributes", "sentiment_valence" = "Emotional Valence\n(Sentiment)",
                           "embedding" = "Word Emeddings", "dass" = "Psychological\nDistress (DASS)")) %>%
  ggplot(aes(x = reorder(category, info_gained), y = info_gained)) +
  geom_bar(stat="identity", fill = "dodgerblue3") +
  coord_flip(clip = "off") +
  scale_y_continuous(expand = c(0, 0), n.breaks = 10, limits = c(0, 35)) +
  labs(
    x = "Independent Variable Categories",
    y = "Average Information Gained (Gini Index)"
  ) +
  theme_bw() +
  theme(text=element_text(family="serif"))
feat_importance_plot

# Export plots
ggsave("data/results/plots/feat_selected_plot.png", plot = feat_selected_plot,
       width = 5.0, height = 4.0)

ggsave("data/results/plots/feat_importance_plot.png", plot = feat_importance_plot,
       width = 5.0, height = 4.0)

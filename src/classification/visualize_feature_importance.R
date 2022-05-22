# Load dependencies
library(tidyverse)

# Import data and arrange
feat_selected <- read_csv("data/results/feature_selection/number_features_selected.csv") %>%
  arrange(desc(percent_used)) %>%
  rename(category = index)

feat_importance <- read_csv("data/results/feature_selection/avg_feature_importance.csv") %>%
  arrange(desc(info_gained))

# Visualize the number of features selected
feat_selected %>%
  mutate(category = recode(category, "clinical_keywords" = "Clinical Keywords", "n_grams" = "n-grams", 
                           "psycholinguistic" = "Psycholinguistic\nAttributes", "sentiment_valence" = "Emotional Valence\n(Sentiment)",
                           "embedding" = "Word Emeddings", "dass" = "Psychological\nDistress (DASS)")) %>%
  mutate(percent_used = round(percent_used, 2)) %>%
  ggplot(aes(x = reorder(category, percent_used), y = percent_used)) +
  geom_bar(stat="identity", fill = "dodgerblue3") +
  coord_flip() +
  scale_y_continuous(expand = c(0, 0)) +
  labs(
    x = "Independent Variable Categories",
    y = "Percent Used in XGBoost Classifier"
  ) +
  theme_bw()

# Visualize the average information gained
feat_importance %>%
  mutate(category = recode(category, "clinical_keywords" = "Clinical Keywords", "n_grams" = "n-grams", 
                           "psycholinguistic" = "Psycholinguistic\nAttributes", "sentiment_valence" = "Emotional Valence\n(Sentiment)",
                           "embedding" = "Word Emeddings", "dass" = "Psychological\nDistress (DASS)")) %>%
  ggplot(aes(x = reorder(category, info_gained), y = info_gained)) +
  geom_bar(stat="identity", fill = "dodgerblue3") +
  coord_flip() +
  scale_y_continuous(expand = c(0, 0), n.breaks = 10, limits = c(0, 35)) +
  labs(
    x = "Independent Variable Categories",
    y = "Average Information Gained (Gini Index)"
  ) +
  theme_bw()

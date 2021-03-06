library(dplyr)
library(ggplot2)
library(foreach)
library(tidyverse)
library(data.table)

setwd("/Users/tingfengx/desktop/Team-ZeroLoss/data/ieee_data")
full_data = read.csv("corona_tweets_01.csv", sep = ',', header = F)
pruned_data = sample_n(full_data, size=10000)
pruned_data$index = rep(1, nrow(pruned_data))
for (i in 2:9) {
  file_name = paste0("corona_tweets_0", i, ".csv")
  temp_data = read.csv(file_name, sep = ',', header = F)
  temp_pruned = sample_n(temp_data, size=10000)
  temp_pruned$index = rep(i, nrow(temp_pruned))
  pruned_data = rbind(pruned_data, temp_pruned)
}
for (i in 10:74) {
  file_name = paste0("corona_tweets_", i, ".csv")
  temp_data = read.csv(file_name, sep = ',', header = F)
  temp_pruned = sample_n(temp_data, size=10000)
  temp_pruned$index = rep(i, nrow(temp_pruned))
  pruned_data = rbind(pruned_data, temp_pruned)
}

pruned_data$V1 = format(pruned_data$V1, scientific = F)
colnames(pruned_data) = c("tweet_id", "sentiment_score", "index")
write.table(pruned_data, file="../hydrated_data_ieee/pruned_data.csv", row.names=FALSE, sep=",", quote=FALSE)

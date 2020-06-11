library(dplyr)
library(ggplot2)
library(foreach)
library(tidyverse)
library(data.table)
library(viridis)

setwd("~/Desktop/Team-ZeroLoss/data/hydrated_data_ieee/final_data")

data = read.csv("shallow_processed_with_prediction.csv")
data$date = gsub("Mar ", "03-", data$date)
data$date = gsub("Apr ", "04-", data$date)
data$date = gsub("May ", "05-", data$date)
data$date = gsub("Jun ", "06-", data$date)

data = data %>% mutate(textblob_hard = case_when(sentiment_score > 0 ~ 2,
                                          sentiment_score == 0 ~ 1,
                                          sentiment_score < 0 ~ 0))

my_data = aggregate(x = data$textblob_hard, by=list(data$date), FUN=mean)
colnames(my_data) = c("date", "score")

setwd("~/Desktop/Team-ZeroLoss/result")
theme_set(theme_bw())
gg <- ggplot(data=my_data, aes(x=date, y=score, group=1)) + 
  geom_line(size=0.4, color="black")+
  geom_point(size=2, aes(color=score)) +
  scale_color_viridis(option = "D") + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  labs(title="Average TextBlob hard Scores by Dates") +
  ylab("mean of TextBlob hard scores")

plot(gg)

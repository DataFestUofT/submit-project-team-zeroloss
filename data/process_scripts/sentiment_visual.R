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
data$year = substring(data$user.created_at, 26, 30)

data = data %>% mutate(predicted_sentiment_indice = case_when(predicted_sentiment_indice == 0 ~ "Negative",
                                                              predicted_sentiment_indice == 1 ~ "Neutral",
                                                              predicted_sentiment_indice == 2 ~ "Positive"))

data = data %>% mutate(textblob_hard = case_when(sentiment_score > 0 ~ 2,
                                          sentiment_score == 0 ~ 1,
                                          sentiment_score < 0 ~ 0))

setwd("~/Desktop/Team-ZeroLoss/result")


# time series plot of sentiment of each date
my_data = aggregate(x = data$predicted_sentiment_indice, by=list(data$date), FUN=mean)
colnames(my_data) = c("date", "score")

theme_set(theme_bw())
gg <- ggplot(data=my_data, aes(x=date, y=score, group=1)) + 
  geom_line(size=0.4, color="black")+
  geom_point(size=2, aes(color=score)) +
  scale_color_viridis(option = "D") + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  labs(title="Average Predicted Scores by Dates") +
  ylab("mean of predicted scores")
plot(gg)

# retweet count
data$retweet_range <- cut(data$retweet_count, breaks=c(-1, 10, 100, 500, 1000, 5000, 10000, 50000, 100000, 200000, Inf))
retweet_stats = data %>% group_by(retweet_range, predicted_sentiment_indice) %>% summarise(n=n())  %>% mutate(freq=n/sum(n))

theme_set(theme_classic())
gg2 <- ggplot(data=retweet_stats, aes(x=retweet_range, y=freq, group=predicted_sentiment_indice, fill=predicted_sentiment_indice)) + 
  geom_bar(stat="identity", position="dodge") + 
  theme(axis.text.x = element_text(angle=65, vjust=0.6))+
  labs(title="Sentiment by Retweet Count Ranges") + 
  guides(fill=guide_legend("predicted sentiment")) + 
  ylab("class fraction within the retweet range")
plot(gg2)

# default profile
profile_stats = data %>% group_by(user.default_profile, predicted_sentiment_indice) %>% summarise(n=n()) %>% mutate(freq=n/sum(n))

theme_set(theme_classic())
gg3 <- ggplot(data=profile_stats, aes(x=user.default_profile, y=freq, group=predicted_sentiment_indice, fill=predicted_sentiment_indice)) + 
  geom_bar(stat="identity", position="dodge") + 
  theme(axis.text.x = element_text(angle=65, vjust=0.6))+
  labs(title="Sentiment by user.default_profile") + 
  guides(fill=guide_legend("predicted sentiment")) + 
  ylab("class fraction within the group")
plot(gg3)

# created_at
year_stats = data %>% group_by(year, predicted_sentiment_indice) %>% summarise(n=n()) %>% mutate(freq=n/sum(n))
year_stats = year_stats[-1,][-1,]

theme_set(theme_classic())
gg4 <- ggplot(data=year_stats, aes(x=year, y=freq, group=predicted_sentiment_indice, fill=predicted_sentiment_indice)) + 
  geom_bar(stat="identity", position="dodge") + 
  theme(axis.text.x = element_text(angle=65, vjust=0.6))+
  labs(title="Sentiment by user.created_at") + 
  guides(fill=guide_legend("predicted sentiment")) + 
  ylab("class fraction within the group")
plot(gg4)

# followers_count
data$follower_range <- cut(data$retweet_count, breaks=c(-1, 500, 5000, 20000, 50000, 200000, 500000, Inf))
follower_stats = data %>% group_by(follower_range, predicted_sentiment_indice) %>% summarise(n=n())  %>% mutate(freq=n/sum(n))

theme_set(theme_classic())
gg5 <- ggplot(data=follower_stats, aes(x=follower_range, y=freq, group=predicted_sentiment_indice, fill=predicted_sentiment_indice)) + 
  geom_bar(stat="identity", position="dodge") + 
  theme(axis.text.x = element_text(angle=65, vjust=0.6))+
  labs(title="Sentiment by user.folowers_count") + 
  guides(fill=guide_legend("predicted sentiment")) + 
  ylab("class fraction within the group")
plot(gg5)

# username_contains_emoji
user_stats = data %>% group_by(username_contains_emoji, predicted_sentiment_indice) %>% summarise(n=n())  %>% mutate(freq=n/sum(n))

theme_set(theme_classic())
gg6 <- ggplot(data=user_stats, aes(x=username_contains_emoji, y=freq, group=predicted_sentiment_indice, fill=predicted_sentiment_indice)) + 
  geom_bar(stat="identity", position="dodge") + 
  theme(axis.text.x = element_text(angle=65, vjust=0.6))+
  labs(title="Sentiment by whether username constains emoji") + 
  guides(fill=guide_legend("predicted sentiment")) + 
  ylab("class fraction within the group")
plot(gg6)

# user.statuses_count
data$status_range <- cut(data$user.statuses_count, breaks=c(-1, 500, 5000, 20000, 50000, 200000, 500000, Inf))
user_stats3 = data %>% group_by(status_range, predicted_sentiment_indice) %>% summarise(n=n())  %>% mutate(freq=n/sum(n))

theme_set(theme_classic())
gg7 <- ggplot(data=user_stats3, aes(x=status_range, y=freq, group=predicted_sentiment_indice, fill=predicted_sentiment_indice)) + 
  geom_bar(stat="identity", position="dodge") + 
  theme(axis.text.x = element_text(angle=65, vjust=0.6))+
  labs(title="Sentiment user.statuses_count") + 
  guides(fill=guide_legend("predicted sentiment")) + 
  ylab("class fraction within the group")
plot(gg7)


  
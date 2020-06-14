library(dplyr)
library(ggplot2)
library(foreach)
library(tidyverse)
library(data.table)

setwd("~/Desktop/Team-ZeroLoss/result/")



data <- read.csv("results.csv")
data = data %>% mutate(predicted_sentiment = case_when(predicted_sentiment == 0 ~ "Negative",
                                                       predicted_sentiment == 1 ~ "Neutral",
                                                       predicted_sentiment == 2 ~ "Positive"))
data$loc = rep(NA, nrow(data))
ca_list = c("CA", "California", "LA", "Los Angeles", "San Jose", "Bay Area", "San Diego", "Santa Clara", "San Francisco")
ca = sapply(ca_list, grepl, data$user.location)
data$ca = apply(ca, 1, any)

ny_list = c("NY", "New York", "Manhattan", "Queens", "Long Island")
ny = sapply(ny_list, grepl, data$user.location)
data$ny = apply(ny, 1, any)
data = data %>% mutate(loc = case_when(ny==T ~ "NY",
                                       ca==T ~ "CA",
                                       T ~ "Other"))

# plotting
stat = data %>% group_by(data$loc, predicted_sentiment) %>% summarise(n=n())  %>% mutate(freq=n/sum(n))

theme_set(theme_gray())
gg7 <- ggplot(data=stat, aes(x=`data$loc`, y=freq, group=predicted_sentiment, fill=predicted_sentiment)) + 
  geom_bar(stat="identity", position="dodge") + 
  theme(axis.text.x = element_text(angle=65, vjust=0.6))+
  labs(title="Sentiment cross locations") + 
  guides(fill=guide_legend("predicted sentiment")) + 
  ylab("class fraction within the group")
plot(gg7)

library(dplyr)
library(ggplot2)
library(foreach)
library(tidyverse)
library(data.table)
library(jsonlite)

setwd("/Users/Raina/desktop/Team-ZeroLoss/data/hydrated_data_ieee")
data1 = read.csv("../semeval2017/twitter-2013train-A.tsv", sep="\t", header=F)
data2 = read.csv("../semeval2017/twitter-2015train-A.tsv", sep="\t", header=F)
data3 = read.csv("../semeval2017/twitter-2016train-A.tsv", sep="\t", header=F)

process <- function(data){
  colnames(data) = c("tweet_id", "sentiment", "text")
  # remove urls from text
  data$text = gsub(" ?(f|ht)(tp)(s?)(://)(.*)[.|/](.*)","", data$text)
  
  # remove retweet RT @..: text
  data$text = gsub("RT @.*: ", "", data$text)
  
  # replace @username with [someone], square brackets denote it's "token" nature
  data$text = gsub("@\\w+", "[someone]", data$text)
  
  # replace utf encoded symbols
  data$text = gsub("\\\\u2019", "'", data$text)
  data$text = gsub("\\\\u002c", ",", data$text)
  data$text = gsub('\\\\\"\"', '"', data$text)
  data$text = gsub("\\", "", data$text, fixed=TRUE)
  
  data = filter(data, text != "")
  return(data)
}

final_data1 = process(data1)
final_data2 = process(data2)
final_data3 = process(data3)

fwrite(final_data1, file ="../training_data/data1.csv")
fwrite(final_data2, file ="../training_data/data2.csv")
fwrite(final_data3, file ="../training_data/data3.csv")




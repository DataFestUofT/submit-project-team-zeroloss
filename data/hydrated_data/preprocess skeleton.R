library(dplyr)
library(ggplot2)
library(foreach)
library(tidyverse)
library(data.table)

full_dataset_clean = read.csv("../full_dataset_clean/full_dataset_clean.tsv", sep = '\t', header = TRUE)
full_dataset_clean$month = substring(full_dataset_clean$date, 6, 7)
data_by_month = filter(full_dataset_clean, month != "01")
my_data = data_by_month %>% group_by(date) %>% sample_n(size=2000)
droppings = c("month")
my_data = my_data[, !(colnames(my_data) %in% droppings)]
write.table(my_data, file="./pruned_data.tsv", row.names=FALSE, sep="\t", quote=FALSE)

library(dplyr)
library(ggplot2)
library(foreach)
library(tidyverse)
library(data.table)
library(jsonlite)

setwd("/Users/Raina/desktop/Team-ZeroLoss/data/hydrated_data_ieee")
tweet_data <- stream_in(file("hydrated_prune"))
# choose useful variables to keep
keep_vars = c("created_at", "id", "text", "truncated", "user", "place", "retweet_count",
              "favorite_count", "lang", "possibly_sensitive", "withheld_in_countries", 
              "withheld_copyright"
              )
tweet_data = tweet_data[, (colnames(tweet_data) %in% keep_vars)]

# first filter
# 1. keep truncated == FALSE
# 2. keep lang = en
filtered = filter(tweet_data, lang== "en" & truncated == FALSE)

# flatten the nested json objects
flattened = flatten(filtered)

# choose useful variables to keep in flattened data
keep_vars_2 = c("created_at", "id", "text", "truncated", "retweet_count", "favorite_count", "lang", "possibly_sensitive",
                "withheld_in_countries", "withheld_copyright", "user.id", "user.name", "user.screen_name", 
                "user.location", "user.description", "user.followers_count", "user.friends_count", "user.listed_count",
                "user.created_at", "user.favourites_count", "user.statuses_count", "user.default_profile_image",
                "user.default_profile", "user.withheld_in_countries", "place.name", "place.full_name", "place.country", 
                "place.country_code", "place.place_type"
              )
flattened = flattened[, (colnames(flattened) %in% keep_vars_2)]

######### restrict the location within the US #########

# remove tweets without any location info
location_data = filter(flattened, user.location != "" | !is.na(place.name) | !is.na(place.full_name) |
                         !is.na(place.country) | !is.na(place.country_code) | !is.na(place.place_type))

# keep all tweets that user purports to be within the US
loc_file = "us_codes.csv"
us_locs = read.delim(loc_file, header=F, sep="\n")[[1]]
us_info = sapply(us_locs, grepl, location_data$user.location)
us_info_single = apply(us_info, 1, any)
location_data$us_info_single = us_info_single

# keep all tweets whose objective place is within the US
us_info_2 = sapply(us_locs, grepl, location_data$place.full_name)
us_info_single_2 = apply(us_info_2, 1, any)
location_data$us_info_single_2 = us_info_single_2

usa_data = filter(location_data, us_info_single | us_info_single_2 | place.country == "United States" | 
                    place.country_code == "US")

# create a date variable
time = strsplit(usa_data$created_at, " ")
time_df = as.data.frame(do.call(rbind, time))
date = paste(time_df$V2, time_df$V3)
usa_data$date = date


####### read in the tweet ids and append the sentiment score to our dataframe ########
tweet_ids <- read.csv("pruned_data.csv", header=T, sep=",", quote = F)
tweet_ids = tweet_ids[, !(colnames(tweet_ids) %in% c("index"))]
tweet_ids = format(tweet_ids, scientific=F)

colnames(usa_data)[2] = "tweet_id"
final_data = merge(usa_data, tweet_ids, by="tweet_id")

####### processing the texts ########

# remove urls from text
final_data$text = gsub(" ?(f|ht)(tp)(s?)(://)(.*)[.|/](.*)","", final_data$text)

# remove retweet RT @..: text
final_data$text = gsub("RT @.*: ", "", final_data$text)

# replace @username with [someone], square brackets denote it's "token" nature
final_data$text = gsub("@\\w+", "[someone]", final_data$text)


###### other trivial things
final_data$tweet_id = format(final_data$tweet_id, scientific = F)
final_data$user.id = format(final_data$user.id, scientific = F)
final_data[final_data == "NULL"] = NA

# drop useless variables
droppings = c("place.place_type", "place.name", "place.full_name", "place.country_code", "place.country" , 
              "us_info_single",  "us_info_single_2", "withheld_copyright", "truncated", "lang", "created_at")
final_data = final_data[, !(colnames(final_data) %in% droppings)]

fwrite(final_data, file ="cleaned_data.csv")


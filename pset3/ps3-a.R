# title:    ps3-a.R
# authors:  nadia lucas and fern ramoutar
# updated:  march 2021

#================================================#
# set up
#================================================#

# packages
library(tidyverse)
library(tigris)
library(sf)
library(RColorBrewer)
library(ggmap)
library(ggthemes)
library(mapview)
library(webshot)
library(ggpubr)
library(Hmisc)
library(knitr)

# set directory
ddir <- "/Users/fernramoutar/Dropbox/github/iobros/pset3"
setwd(ddir)

#================================================#
# question 1
#================================================#

### summary statistics by auction house

# load dataset + gen dummy for ring winners "test"
stamps <-
  read_csv("ps3.csv") %>%
  rename(target = "realisation in final auAtion") %>%
  rename(min_est = "EstDated MinDum") %>%
  rename(max_est = "EstDated MaxDum") %>%
  rename(payment = "Net  Payment") %>%
  group_by(house,lot) %>%
  mutate(maxbid = max(bid)) %>%
  mutate(test = ifelse(target <= maxbid & bid == maxbid, 1, 0)) %>%
  ungroup

# mean + sd of knockout/target bids
house_stamps <- stamps %>%
  group_by(house) %>%
  summarise(house_target=mean(target), house_target_sd = sd(target), 
            house_knockout = mean(bid), house_knockout_sd = sd(bid))

# pct value won 
lot_stamps <- stamps %>%
  mutate(avg_est = (min_est + max_est)/2) %>%
  distinct(lot, .keep_all = TRUE) %>%
  mutate(urgh = test * avg_est) %>% # value if ring person won
  group_by(house) %>%
  summarise(ring_val = sum(urgh), total_val = sum(avg_est)) %>%
  mutate(pct_value = ring_val / total_val)

# pct lots won + total lots 
more_stamps <- stamps %>%
  group_by(house,lot) %>%
  summarise(won=sum(test)) %>%
  mutate(ones = 1) %>%
  group_by(house) %>% 
  summarise(ring_won = sum(won), total_lots = sum(ones)) %>%
  mutate(pct_won = ring_won/total_lots)

# join results
right_join <- merge(house_stamps, lot_stamps, by = "house", all.y = TRUE)
right_join <- merge(right_join, more_stamps, by = "house", all.y = TRUE)

### summary statistics by number of bidders

# mean + sd of knockout/target bids
number_stamps <- stamps %>%
  group_by(house,lot) %>%
  mutate(ringers = max(bidder)) %>%
  group_by(ringers) %>%
  summarise(house_target=mean(target), house_target_sd = sd(target), 
            house_knockout = mean(bid), house_knockout_sd = sd(bid))
  
# # pct lots won + total lots 
bidder_stamps <- stamps %>%
  group_by(house,lot) %>%
  mutate(ringers = max(bidder)) %>%
  distinct(lot, .keep_all = TRUE) %>%
  mutate(ones = 1) %>%
  group_by(ringers) %>%
  summarise(won=sum(test), total = sum(ones)) %>%
  mutate(pct_won = won/total)

### remove all lots where there are more than 2 bidders

redo <- stamps %>%
  group_by(house,lot) %>%
  mutate(ringers = max(bidder)) 
  ungroup

final <- subset(redo, ringers < 3)


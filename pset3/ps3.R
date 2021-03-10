# title:    ps3.R
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

# load dataset
stamps <-
  read_csv("ps3.csv") %>%
  rename(target_price = "realisation in final auAtion") %>%
  rename(ko_bid = "bid") %>%
  rename(est_min_dum = "EstDated MinDum") %>%
  rename(est_max_dum = "EstDated MaxDum") %>%
  rename(payment = "Net  Payment") %>%
  rename(catalog_price = "Aatalog PriAe") %>%
  rename(catalog_du10y = "Aatalog Du10y") %>%
  rename(est_min = "EstDate Min") %>%
  rename(est_max = "EstDate Max") %>%
  rename(grade_min = "Grade Min") %>%
  rename(grade_max = "Grade Max") %>%
  rename(no_grade = "No Grade") %>%
  rename(us = "ExAlusively US") %>%
  rename(no_value = "No Value") %>%
  rename(data_id = "Data NuEer") %>%
  group_by(house,lot,date) %>%
  mutate(ringers =n()) %>% # number of ring participants
  mutate(max_bid = max(ko_bid)) %>% # highest bid in knockout
  mutate(ring_winner = ifelse(target_price <= max_bid & ko_bid == max_bid, 1, 0)) %>% # ring won
  mutate(ring_won = ifelse(target_price <= max_bid, 1, 0)) %>%
  mutate(highest_ringer = ifelse(ko_bid == max_bid, 1, 0)) %>% # highest ring bidder
  mutate(payer = ifelse(payment>0,1,0)) %>% # made side payment
  mutate(receiver = ifelse(payment<0,1,0)) %>% # received side payment
  mutate(avg_est = (est_min + est_max)/2) %>% # average estimated value of lot
  mutate(small_lots = ifelse(avg_est<10000, 1, 0)) %>% # lots below $10,000
  mutate(ones = 1) %>%
  ungroup

# collapse to lot level
lot_level = stamps %>%
  group_by(house, lot, date) %>%
  summarise(target_price = mean(target_price), max_bid = mean(max_bid), avg_est = mean(avg_est), ring_won = mean(ring_won), ones = 1, ringers = mean(ringers))

# summary stats by auction house
table1 <- lot_level %>% 
  mutate(rwv = ifelse(ring_won==1,ring_won * avg_est,0)) %>%
  group_by(house) %>% 
  summarise(house_target = mean(target_price), house_target_sd = sd(target_price), 
            house_knockout = mean(max_bid), house_knockout_sd = sd(max_bid),
            ring_val = sum(rwv), total_val = sum(avg_est), ring_total = sum(ring_won),
            lots_total = sum(ones)) %>%
  mutate(pct_value = ring_val / total_val) %>%
  mutate(pct_won = ring_total/lots_total) %>%
  select(house,house_target,house_target_sd,house_knockout,house_knockout_sd,
         pct_won,pct_value,lots_total) # what is a sale?

# summary stats by number of bidders
table2 <- stamps %>%
  group_by(ringers) %>%
  summarise(ringers_target=mean(target_price), ringers_target_sd = sd(target_price), 
            ringers_knockout = mean(ko_bid), ringers_knockout_sd = sd(ko_bid),
            ring_total=sum(ring_won), lots_total = sum(ones)) %>%
  mutate(pct_won = ring_total/lots_total) %>%
  mutate(lots_total = lots_total/ringers) %>%
  select(ringers,ringers_target,ringers_target_sd,ringers_knockout,ringers_knockout_sd,
         pct_won,lots_total)

#================================================#
# question 2
#================================================#

# knockout outcomes by ring member
table5 <- stamps %>%
  group_by(bidder) %>%
  add_tally() %>%
  rename(n_any = "n") %>%
  mutate(highest_any=sum(highest_ringer)) %>%
  mutate(pctkoany = highest_any/n_any) %>%
  ungroup %>%
  subset(ringers>=2) %>%
  group_by(bidder) %>%
  add_tally() %>%
  mutate(highest=sum(highest_ringer)) %>%
  mutate(pctko = highest/n) %>%
  mutate(paying = sum(payer)) %>%
  mutate(receiving=sum(receiver)) %>%
  mutate(pctpaying = paying/n) %>%
  mutate(pctreceiving = receiving/n) %>%
  summarise(highest_total_any = mean(highest_any), pct_ko_any = mean(pctkoany), 
            highest_total = mean(highest), pct_ko = mean(pctko), 
            pct_paying=mean(pctpaying), pct_receiving = mean(pctreceiving))

# net side payments
ggplot(stamps) + 
  geom_bar(aes(bidder, payment, fill = as.factor(small_lots)), 
           position = "dodge", stat = "summary", fun.y = "mean")

#================================================#
# create structural dataset
#================================================#

ok <- subset(stamps, ringers == 2) 

# auction characteristics 
auctions <- ok %>%
  select(house,date,lot,est_min,est_max,catalog_price,grade_min,grade_max,us,no_value) %>%
  distinct()

# knockout bids
ko <-  ok %>%
  select(house,date,lot,bidder,ko_bid) %>%
  mutate(bidder1 = ifelse(bidder == 1,1,0)) %>%
  mutate(bidder2 = ifelse(bidder == 2,1,0)) %>%
  select(house,date,lot,ko_bid,bidder1,bidder2) %>%
  rename(bid = "ko_bid") %>%
  mutate(target = 0)
  
# target bids
target <- ok %>%
  group_by(house,date,lot) %>%
  summarise(bid = mean(target_price)) %>%
  mutate(target = 1) %>%
  mutate(bidder1 = 0) %>%
  mutate(bidder2 = 0)

# append target + knockout bids
bind <- rbind(ko, target)

# merge bids with auction characteristics
stamps_final <- bind %>%
  inner_join(auctions) %>%
  mutate(lnb = log(bid))

# first stage regression
fit1 <- lm(lnb ~ est_min + est_max + catalog_price + grade_min + grade_max + us 
           + bidder1 + bidder2 + target, data = stamps_final)

# predicted bids
stamps_final$bhat <- fitted(fit1) 
  

pacman::p_load(tidyverse, haven, sjlabelled, jmv)
dat <- read_dta("Data and Preperation/CCES data/cumulative_2006-2020.dta")

new_dat <- dat %>% 
  mutate(voted = ifelse(vv_turnout_gvm == 1, 1, 0)) %>% 
  select(voted, ## collapsed to voter and non-voter
         year,
         gender, 
         age, 
         educ, # 1 - no HS, 2 - HS, 3- some, 4 - 2year, 5- 4year, 6- postGrad, NA- since only 67 lets delete those
         race, 
         citizen, # missingness attributed to years where question wasn't asked, this also leaves question if we should delete these responses because they cannot vote
         religion, # lets make NAs as nothing for 2006 leave out
         marstat, # make missings single
         vv_party_prm, #available 2012, 2014, 2016, 2018, simplify to DEM, REP, and OTHER
         pid3_leaner, # keep this one simple and add granularity with ideology
         ideo5,
         employ,
         economy_retro, # change scale to put not sure with about the same
         approval_pres, # for all the approvals put the neigther option in the middle of the scale
         approval_rep,
         approval_gov,
         approval_sen1,
         approval_sen2
    ) %>% 
  filter(year %in% c(2006, 2008, 2010, 2012, 2014, 2016, 2018)) %>%   ## Keep only election years
  mutate(marstat = ifelse(is.na(marstat), 5, marstat),
         economy_retro = ifelse(is.na(economy_retro) | economy_retro == 6, 3, economy_retro),
         approval_pres = case_when(approval_pres %in% c(5, 6) ~ 0,
                                   approval_pres == 1 ~ 2,
                                   approval_pres == 2 ~ 1,
                                   approval_pres == 3 ~ -1,
                                   approval_pres == 4 ~ -2),
         approval_rep = case_when(approval_rep %in% c(5, 6) ~ 0,
                                   approval_rep == 1 ~ 2,
                                   approval_rep == 2 ~ 1,
                                   approval_rep == 3 ~ -1,
                                   approval_rep == 4 ~ -2),
         approval_gov = case_when(approval_gov %in% c(5, 6) ~ 0,
                                   approval_gov == 1 ~ 2,
                                   approval_gov == 2 ~ 1,
                                   approval_gov == 3 ~ -1,
                                   approval_gov == 4 ~ -2),
         approval_sen1 = case_when(approval_sen1 %in% c(5, 6) ~ 0,
                                   approval_sen1 == 1 ~ 2,
                                   approval_sen1 == 2 ~ 1,
                                   approval_sen1 == 3 ~ -1,
                                   approval_sen1 == 4 ~ -2),
         approval_sen2 = case_when(approval_sen2 %in% c(5, 6) ~ 0,
                                   approval_sen2 == 1 ~ 2,
                                   approval_sen2 == 2 ~ 1,
                                   approval_sen2 == 3 ~ -1,
                                   approval_sen2 == 4 ~ -2)
         )


setwd("~/Research/Voter_Predictions/Data and Preperation/CCES data")

write.csv(new_dat, "clean_cces_data.csv")

### Categorical variables ###
# gender
# educ
# race
# citizen
# religion
# marstat
# vv_party_prm
# pid3_leaner
# employ


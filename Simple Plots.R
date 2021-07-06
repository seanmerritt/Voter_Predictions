pacman::p_load(tidyverse, haven, sjlabelled, jmv)
dat <- read_dta("Data and Preperation/CCES data/cumulative_2006-2020.dta")

setwd('C:\Users\seanm\Documents\Research\Voter_Predictions\Plots')

dat %>% 
  filter(approval_pres < 5, year %in% c(2006,2010,2014,2018), ideo5 < 6) %>%
  mutate(turnout = ifelse(vv_turnout_gvm == 1, "Yes", "No"),
         party = case_when(ideo5 %in% c(1,2) ~ "Liberal",
                           ideo5 == 3 ~ "Moderate",
                           ideo5 %in% c(4,5) ~ "Conservative"),
         approval = ifelse(approval_pres >= 3, "Dis.", "App.")) %>% 
  group_by(year, approval, party, turnout) %>% 
  summarize(count = n()) %>% 
  ggplot(aes(fill = turnout, x = approval, y = count))+
  geom_bar(stat = 'identity', position = "dodge")+
  facet_grid(vars(party), vars(year))+
  labs(x = "Approval", fill = "Voted")+
  theme_classic()+
  theme(legend.position = "bottom")

ggsave("Voted_and_approval_plot_1.jpeg")

dat %>% 
  filter(approval_pres < 5, year %in% c(2008,2012,2016), ideo5 < 6) %>%
  mutate(turnout = ifelse(vv_turnout_gvm == 1, "Yes", "No"),
         party = case_when(ideo5 %in% c(1,2) ~ "Liberal",
                           ideo5 == 3 ~ "Moderate",
                           ideo5 %in% c(4,5) ~ "Conservative"),
         approval = ifelse(approval_pres >= 3, "Dis.", "App.")) %>% 
  group_by(year, approval, party, turnout) %>% 
  summarize(count = n()) %>% 
  ggplot(aes(fill = turnout, x = approval, y = count))+
  geom_bar(stat = 'identity', position = "dodge")+
  facet_grid(vars(party), vars(year))+
  theme_classic()+
  labs(x = "Approval", fill = "Voted")+
  theme(legend.position = "bottom")

ggsave("Voted_and_approval_plot_2.jpeg")

dat %>% 
  filter(approval_pres < 5, year %in% c(2006,2010,2014,2018)) %>%
  mutate(turnout = ifelse(vv_turnout_gvm == 1, "Yes", "No"),
         party = case_when(pid3 == 1 ~ "Democrat",
                           pid3 == 2 ~ "Republican",
                           pid3 %in% c(3,4,5) | is.na(pid3) ~ "Other"),
         approval = ifelse(approval_pres >= 3, "Dis.", "App.")) %>% 
  group_by(year, approval, party, turnout) %>% 
  summarize(count = n()) %>% 
  ggplot(aes(fill = turnout, x = approval, y = count))+
  geom_bar(stat = 'identity', position = "dodge")+
  facet_grid(vars(party), vars(year))+
  labs(x = "Approval", fill = "Voted")+
  theme_classic()+
  theme(legend.position = "bottom")

ggsave("Voted_and_approval_partisan_1.jpeg")


dat %>% 
  filter(approval_pres < 5, year %in%  c(2008,2012,2016) )%>%
  mutate(turnout = ifelse(vv_turnout_gvm == 1, "Yes", "No"),
         party = case_when(pid3 == 1 ~ "Democrat",
                           pid3 == 2 ~ "Republican",
                           pid3 %in% c(3,4,5) | is.na(pid3) ~ "Other"),
         approval = ifelse(approval_pres >= 3, "Dis.", "App.")) %>% 
  group_by(year, approval, party, turnout) %>% 
  summarize(count = n()) %>% 
  ggplot(aes(fill = turnout, x = approval, y = count))+
  geom_bar(stat = 'identity', position = "dodge")+
  facet_grid(vars(party), vars(year))+
  labs(x = "Approval", fill = "Voted")+
  theme_classic()+
  theme(legend.position = "bottom")

ggsave("Voted_and_approval_partisan_2.jpeg")

dat %>% 
  filter(approval_pres < 5, year %in%  c(2006,2010,2014,2018))%>%
  unlabel() %>% 
  mutate(turnout = ifelse(vv_turnout_gvm == 1, "Yes", "No"),
         party = case_when(pid7 == 8 | is.na(pid7) ~ 4,
                           TRUE ~ pid7),
         approval = ifelse(approval_pres >= 3, "Dis.", "App.")) %>% 
  group_by(year, approval, party, turnout) %>% 
  summarize(count = n()) %>% 
  ggplot(aes(fill = turnout, x = approval, y = count))+
  geom_bar(stat = 'identity', position = "dodge")+
  facet_grid(vars(party), vars(year))+
  labs(x = "Approval", fill = "Voted")+
  theme_classic()+
  theme(legend.position = "bottom")

ggsave("Voted_and_approval_partisan_7point_1.jpeg")

dat %>% 
  filter(approval_pres < 5, year %in%  c(2008,2012,2016) )%>%
  unlabel() %>% 
  mutate(turnout = ifelse(vv_turnout_gvm == 1, "Yes", "No"),
         party = case_when(pid7 == 8 | is.na(pid7) ~ 4,
                           TRUE ~ pid7),
         approval = ifelse(approval_pres >= 3, "Dis.", "App.")) %>% 
  group_by(year, approval, party, turnout) %>% 
  summarize(count = n()) %>% 
  ggplot(aes(fill = turnout, x = approval, y = count))+
  geom_bar(stat = 'identity', position = "dodge")+
  facet_grid(vars(party), vars(year))+
  labs(x = "Approval", fill = "Voted")+
  theme_classic()+
  theme(legend.position = "bottom")

ggsave("Voted_and_approval_partisan_7point_2.jpeg")

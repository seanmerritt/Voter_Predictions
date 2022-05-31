
dat %>% 
  filter(approval_pres < 5, year %in% c(2006, 2008, 2010, 2012, 2014, 2016, 2018), ideo5 < 6) %>%
  mutate(turnout = ifelse(vv_turnout_gvm == 1, 1,0)) %>% 
  jmv::logRegBin(dep = turnout, 
                 covs = vars(approval_pres, ideo5, year),
                 blocks = list(list('approval_pres', 'ideo5', 'year')),
                 refLevels = list(list(var = "turnout", ref = "0")),
                 acc = T,
                 class = T
  )

for(y in c(2006, 2008, 2010, 2012, 2014, 2016, 2018)){
  model_dat <- new_dat %>% 
    filter(approval_pres < 5, year == y, ideo5 < 6) %>%
    na.omit() %>% 
    mutate(turnout = ifelse(vv_turnout_gvm == 1, 1,0)) %>% 
    jmv::logRegBin(dep = turnout, 
                   covs = vars(approval_pres, ideo5),
                   blocks = list(list('approval_pres', 'ideo5')),
                   refLevels = list(list(var = "turnout", ref = "0")),
                   acc = T,
                   class = T
    )
  print(model)
  
}

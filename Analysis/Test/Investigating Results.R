new_dat %>% 
  na.omit() %>%
  ggplot(aes(x = age, fill = factor(voted)))+
  geom_bar(position = "dodge")

dat_omit <- new_dat %>% 
  na.omit()

dat_omit %>% 
  unlabel() %>%
  mutate(y_pred = ifelse(vv_party_prm == "1", 0, 1),
         y_pred = ifelse(y_pred == 0 & age > 45 , 1, y_pred)) %>%
  mutate(y_accuracy = ifelse(y_pred == voted,1,0)) %>% 
  select(y_pred,y_accuracy) %>% 
  summarize(accuracy = sum(y_accuracy)/n())
  
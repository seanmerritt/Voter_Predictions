library(tidyverse)
library(MASS)

survey_2019 <- read_csv("C - Research/NonVoters/survey_2019.csv")


dat <- survey_2019 %>% 
  filter(!is.na(turnout16_2016),
         turnout16_2016 <= 2, 
         helpful_people_2016 < 9,
         trust_people_2016 < 9, 
         wealth_2016 <9,   
         values_culture_2016  < 9, 
         US_respect_2016 < 9,
         fair_people_2016 <9, 
         gender_baseline < 3,
         race_baseline < 9, 
         educ_baseline < 7,
         ideo5_baseline < 7,
         ideo5_2016 < 7) %>% 
  dplyr::select(turnout16_2016, 
                gender_baseline, 
                race_baseline, 
                educ_baseline, 
                RIGGED_SYSTEM_1_2016:fair_people_2016, 
                ideo5_2016, 
                ideo5_baseline,
                imiss_a_2016:imiss_y_2016, 
                ambornin_2016:amdiverse_2016, 
                pew_religimp_2016) %>% 
  mutate(voted = ifelse(turnout16_2016 == 2,0,1),
         ideo5_baseline = ifelse(ideo5_baseline == 6, 3, ideo5_baseline),
         ideo5_2016 = ifelse(ideo5_2016 == 6, 3, ideo5_2016)) %>% 
  rowwise() %>% 
  mutate(ideology = mean(ideo5_2016:ideo5_baseline)) %>% 
  dplyr::select(RIGGED_SYSTEM_1_2016:voted, gender_baseline, race_baseline, educ_baseline, ideology )

dat %>% 
  jmv::descriptives()


full.model <- glm(voted == 1 ~., dat, family = binomial)
step.model <- full.model %>% stepAIC(trace = FALSE)

summary(step.model)


dat %>% 
  jmv::logRegBin(dep = voted ,
              covs = vars(RIGGED_SYSTEM_1_2016, RIGGED_SYSTEM_4_2016, RIGGED_SYSTEM_6_2016, econtrend_2016,Americatrend_2016, futuretrend_2016 ,wealth_2016 , values_culture_2016, US_respect_2016, trust_people_2016, helpful_people_2016, gender_baseline, race_baseline, educ_baseline ),
              blocks = list( list( 'RIGGED_SYSTEM_1_2016', 'RIGGED_SYSTEM_4_2016', 'RIGGED_SYSTEM_6_2016', 'econtrend_2016','Americatrend_2016', 'futuretrend_2016' ,'wealth_2016' , 'values_culture_2016', 'US_respect_2016', 'trust_people_2016', 'helpful_people_2016', 'gender_baseline', 'race_baseline' , 'educ_baseline'  )),
              refLevels = list(list(var = "voted", ref = "0" )), 
              acc = T,
              class = T,
              modelTest = T, 
              OR =T, 
              collin = T,
              #stdEst = T
              )
## SMOTE data set, figure out which variables to use
              
write.csv(dat,"voting_data.csv")

dat %>% 
  dplyr::select(RIGGED_SYSTEM_1_2016:pew_religimp_2016, ideology) %>% 
  EGAnet::EGA()


model = c(251,   81, 400, 7046)
base = c(0,332, 0, 7446)
chisq.test(model,base)

dat %>% na.omit()

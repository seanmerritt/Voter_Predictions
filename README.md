# Voter_Predictions
## Motivation
While voting is a big part of a democratic society, a lot of adults don't vote. To better understand this, I examined voting data to find the best predictors of voting status. 

## Data
This project used two different data sets on voting between 2010 and 2020. The first data set (ERS) was an imbalanced data containing 97% of observations as voted. The other data set (CCES data) contains similar information but is more representative of the population and the voter status has been verified after each election.  The ERS contains ideological data that the other doesn't which is why both data sets were used. 

## Method
After cleaning the data, I used the imblearn package to use a SMOTE technique on the ERS data. Given that it is imbalanced machine learning techniques will have a difficult time learning from the data. SMOTE is able to synthetically up sample to balance the data. I then split both data sets into a test and training set and trained a gradient boosted tree to predict voters. The hyperparameters were used using 5-fold cross validation. Given that each person was only measured once, I split up the data for each year to account for the differences accross years. Accuracy was measured using area under the curve (AUC).

## Results
I was able to predict voters with 97%-99% accuracy (AUC) in the ERS data. However, I was only able to predict voters 64%-80% (AUC) of the time with the CCES data. Given the lack of replicability accross the data  and the questionability of the ERS data, I was not able to make any strong conclusions about what predicts whether someone will vote. However, here is some considerations for the important factors that predict voter status: their belief that the system of voting is rigged, their party leaning, age, race, education, and their approval of the president.

## Implications
Most of the top predictors of voter status are not influentiable. However, one key variable (belief the system is rigged) can be. Therefore, we reccomend that more needs to be done to increase voters trust in the system. 

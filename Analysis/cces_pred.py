import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier, AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.model_selection import train_test_split

def ensemble_params(X_train, y_train, CV = 5, boosted = True):
    ## Logistic regression C
    log = LogisticRegression( max_iter = 1000)
    params = {'C': [1.0,100.0]}
    grid_log = GridSearchCV(log, params, cv = CV)
    grid_log.fit(X_train, y_train)
    log_C = grid_log.best_params_['C']
    print("Log complete")
    ## K-neirest neighbor
    KNN = KNeighborsClassifier()
    params = {'n_neighbors': [3,4]}
    grid_KNN = GridSearchCV(KNN, params, cv = CV)
    grid_KNN.fit(X_train, y_train)
    k_neighbors = grid_KNN.best_params_['n_neighbors']
    print("knn complete")
    ## Decision tree max depth
    tree = DecisionTreeClassifier()
    params = {'max_depth': [3,4,5,6,7,8,9,10]}
    grid_tree = GridSearchCV(tree, params, cv = CV)
    grid_tree.fit(X_train, y_train)
    tree_max_depth = grid_tree.best_params_['max_depth']
    print("tree complete")
    ## Random forest max depth
    rnd = RandomForestClassifier()
    params = {'max_depth': [3,4,5,6,7,8,9,10]}
    grid_rnd = GridSearchCV(rnd, params, cv = CV)
    grid_rnd.fit(X_train, y_train)
    rnd_max_depth = grid_rnd.best_params_['max_depth']
    print("rnd complete")
    ## Support vector machine
    svm = SVC(probability= True)
    params = {'C':[1,2,3,4,5]} 
    grid_svm = GridSearchCV(svm, params, cv = CV)
    grid_svm.fit(X_train, y_train)
    svm_C = grid_svm.best_params_['C']
    print("svm complete")
    if boosted:
        ## Gradient boosting
        gbc = GradientBoostingClassifier( learning_rate=0.1)
        params = {'n_estimators': [50,75,100,125,150],
        'max_depth':[3,4,5,6,7,8,9,10,11,12]} 
        grid_gbc = GridSearchCV(gbc, params, cv = CV)
        grid_gbc.fit(X_train, y_train)
        gbc_n_estimators = grid_gbc.best_params_['n_estimators']
        gbc_max_depth = grid_gbc.best_params_['max_depth']
        print("GBC complete")
        ## ADA boosting
        ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=10), algorithm= 'SAMME.R', learning_rate=0.1)
        params = {'n_estimators': [50,75,100,125,150]} 
        grid_ada = GridSearchCV(ada, params, cv = CV)
        grid_ada.fit(X_train, y_train)
        ada_n_estimators = grid_ada.best_params_['n_estimators']
        print("ADA complete")
    else:
        gbc_n_estimators = 0 
        gbc_max_depth = 0
        ada_n_estimators = 0

    return {'log_C':log_C, 'k_neighbors':k_neighbors, 'tree_max_depth':tree_max_depth, 
            'rnd_max_depth':rnd_max_depth, 'svm_C':svm_C, 'gbc_n_estimators':gbc_n_estimators, 
            'gbc_max_depth':gbc_max_depth, 'ada_n_estimators':ada_n_estimators}

data = pd.read_csv(r'C:\Users\seanm\Documents\Research\Voter_Predictions\Data and Preperation\CCES data\clean_cces_data.csv')
data = data.dropna()
y = data['voted']
X = data.iloc[:, 2:]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= .25, random_state= 123)

best_params = ensemble_params(X_train, y_train, CV = 2, boosted = False)
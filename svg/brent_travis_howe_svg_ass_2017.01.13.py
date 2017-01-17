import data_clean
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.grid_search import GridSearchCV
from sklearn.metrics.scorer import make_scorer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


def rf_go(X_train, y_train):
    print "classifier is beginning...\n"
    rf = RandomForestClassifier()

    # param_dict = {'n_estimators': [10, 50, 100, 500],
    #     'max_features': ['auto', 5, 10, 20, 25]}
    # 500 and 25
    param_dict = {'n_estimators': [10, 45, 50, 55],
        'max_features': ['auto', 25, 35, 45, 55]}
    gsCV_rf = GridSearchCV(rf, param_dict, n_jobs = -1, scoring='roc_auc')
    gsCV_rf.fit(X_train, y_train)

    print gsCV_rf.best_params_
    print gsCV_rf.best_score_

    return



def feature_importance(X_train, y_train, n_est, max_feat):
    print 'feature_importance happening...'
    rf = RandomForestClassifier(n_estimators=n_est, max_features=max_feat)
    rf.fit(X_train, y_train)
    importances = rf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(X_train.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X_train.shape[1]), importances[indices], color="r", yerr=std[indices], align="center")
    plt.xticks(range(X_train.shape[1]), indices)
    plt.xlim([-1, X_train.shape[1]])
    plt.show()



def holdout_predict(X_test, n_est, max_feat):
    rf = RandomForestClassifier(n_estimators=n_est, max_features=max_feat)
    rf.fit(X_train, y_train)

    holdout = pd.read_csv('/Users/brenthowe/datascience/data sets/svg/holdout_.csv')
    holdout['predictions'] = rf.predict(X_test)

    holdout.to_csv('/Users/brenthowe/datascience/data sets/svg/holdout_prediction.csv')



if __name__=="__main__":
    X_train, y_train = data_clean.data_clean('/Users/brenthowe/datascience/data sets/svg/train_test_.csv') # preprocessing is done in the file named data_clean.py
    # n_est, m_feat = rf_go(X_train, y_train)
    n_est = 25; m_feat = 55
    feature_importance(X_train, y_train, n_est, m_feat)

    # X_test = data_clean.data_clean('/Users/brenthowe/datascience/data sets/svg/holdout_.csv', train=False)
    # holdout_predict(X_train, y_train, X_test, n_est, m_feat)

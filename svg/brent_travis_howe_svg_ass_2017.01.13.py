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

    param_dict = {'n_estimators': [10, 45, 50, 55],
        'max_features': ['auto', 25, 35, 45, 55]}
    # param_dict = {'n_estimators': [55],
    #     'max_features': [25]}
    gsCV_rf = GridSearchCV(rf, param_dict, n_jobs = -1, scoring='roc_auc')
    gsCV_rf.fit(X_train, y_train)

    print gsCV_rf.best_params_
    print gsCV_rf.best_score_

    return gsCV_rf



def holdout_predict(X_test, rf_obj):
    print '\nholdout_predict now...\n'
    holdout = pd.read_csv('/Users/brenthowe/datascience/data sets/svg/holdout_.csv')
    holdout['prediction_proba'] = rf_obj.predict_proba(X_test)[:,1]
    holdout.to_csv('/Users/brenthowe/datascience/data sets/svg/holdout_predict.csv')



def feature_importance(rf):
    print 'feature_importance happening...'
    # rf.fit(X_train, y_train)
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
    plt.bar(range(15), importances[indices[:15]], color="r", yerr=std[indices[:15]], align="center")
    plt.xticks(range(15), indices[:15])
    plt.xlim([-1, 15])
    # plt.show()
    plt.savefig('feature_importances.png')




if __name__=="__main__":
    X_train, y_train, X_test = data_clean.data_clean('/Users/brenthowe/datascience/data sets/svg/train_test_.csv', '/Users/brenthowe/datascience/data sets/svg/holdout_.csv') # preprocessing is done in the file named data_clean.py
    rf_obj = rf_go(X_train, y_train)
    holdout_predict(X_test, rf_obj)

    rf = RandomForestClassifier(n_estimators=25, max_features=55)
    rf.fit(X_train, y_train)
    feature_importance(rf)

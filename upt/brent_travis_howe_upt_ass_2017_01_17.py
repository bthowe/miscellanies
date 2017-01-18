import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.grid_search import GridSearchCV
from sklearn.metrics.scorer import make_scorer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


#Load and preprocess the data, including dealing with missing observations and encoding categorical variables
def load_data(filename, train=True):
    print "load_data...\n"
    df = pd.read_csv(filename)
    le = LabelEncoder()

    #### custAge
    df['custAge'].fillna(df.custAge.median(), inplace=True)

    #### profession
    le.fit(df['profession'])
    df['profession'] = le.transform(df['profession'])

    #### marital
    le.fit(df['marital'])
    df['marital'] = le.transform(df['marital'])

    #### schooling ...nans are treated as separate category
    le.fit(df['schooling'])
    df['schooling'] = le.transform(df['schooling'])

    #### default
    le.fit(df['default'])
    df['default'] = le.transform(df['default'])

    #### housing
    le.fit(df['housing'])
    df['housing'] = le.transform(df['housing'])

    #### loan
    le.fit(df['loan'])
    df['loan'] = le.transform(df['loan'])

    #### contact
    le.fit(df['contact'])
    df['contact'] = le.transform(df['contact'])

    #### month
    le.fit(df['month'])
    df['month'] = le.transform(df['month'])

    #### day_of_week ...nans are treated as separate category
    le.fit(df['day_of_week'])
    df['day_of_week'] = le.transform(df['day_of_week'])

    #### campaign...looks fine

    #### pdays...conditional on having been contacted the quartiles are 3, 6, and 7
    df['pdays_bin'] = 0
    df.ix[df.pdays <= 6, 'pdays_bin'] = 1
    df.ix[(df.pdays > 6) & (df.pdays<999), 'pdays_bin'] = 2
    df.drop(['pdays'], 1, inplace=True)

    #### previous...looks fine

    #### poutcome
    le.fit(df['poutcome'])
    df['poutcome'] = le.transform(df['poutcome'])

    #### emp.var.rate, cons.price.idx, cons.conf.idx, euribor3m, all look fine

    #### nr.employed...I'm assuming this is the number of employees at the insurance company

    #### pmonths...is going to be correlated with pdays
    df.drop(['pmonths'], 1, inplace=True)

    #### pastEmail...looks fine

    #### id
    df.drop(['id'], 1, inplace=True)


    # print df.info()
    if train:
        #### responded
        le.fit(df['responded'])
        df['responded'] = le.transform(df['responded'])

        #### profit....looks fine

        #### dataset conditional on purchase
        # df_c_index = df[df.profit.isnull()].index
        df_r_index = df[df.profit.notnull()].index

        y = df[['responded', 'profit']]
        df.drop(['responded', 'profit'], 1, inplace=True)
        X = df

        enc = OneHotEncoder(categorical_features = [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 19])
        X_enc = pd.DataFrame(enc.fit_transform(X).toarray())

        X_c_enc = X_enc
        X_r_enc = X_enc.iloc[df_r_index]

        y_c = y['responded']
        y_r = y['profit'].iloc[df_r_index]

        return X_c_enc, y_c, X_r_enc, y_r
    else:
        X_c = df
        enc = OneHotEncoder(categorical_features = [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 19])
        X_c_enc = enc.fit_transform(X_c).toarray()
        return X_c_enc


# The first thing I do is predict whether an individual will make a purchase
def respond_classifier(X, y):
    print "\nrespond_classifier...\n"
    rf = RandomForestClassifier()

    param_dict = {'n_estimators': [10, 25, 50, 75, 100],
        'max_features': [ 5, 10, 'auto', 20, 25]}
    gsCV_rf = GridSearchCV(rf, param_dict, n_jobs = -1, scoring='roc_auc')
    gsCV_rf.fit(X, y)

    print gsCV_rf.best_params_
    print gsCV_rf.best_score_

    return gsCV_rf


# Second, I predict, conditional on a purchase, the profit
def profit_regress(X, y):
    print "\nprofit_regress...\n"
    rf = RandomForestRegressor()

    param_dict = {'n_estimators': [10, 25, 50, 75, 100],
        'max_features': [ 5, 10, 'auto', 20, 25]}
    gsCV_rf = GridSearchCV(rf, param_dict, n_jobs = -1)
    gsCV_rf.fit(X, y)

    print gsCV_rf.best_params_
    print gsCV_rf.best_score_

    return gsCV_rf


# This function predicts the profit for the testing data set, given the rule that an individual is not marketed if
def profit_predict(X, c_obj, r_obj):
    print "\nprofit_predict...\n"
    df = pd.DataFrame(X)
    H = pd.DataFrame(df.index, columns = ['index'])

    ind = np.where(c_obj.predict(X)==1)

    poffer = r_obj.predict(df.iloc[ind])
    offer = np.where(poffer>30, 1, 0)
    # poffer = np.where(pred>30, pred-30, 0)

    a = np.array([ind[0], offer, poffer]).T

    ind_offer = pd.DataFrame(a, columns = ['index', 'offer', 'profit'])

    H = H.merge(ind_offer, how='left', on='index')
    H['offer'].fillna(0, inplace=True)
    H['profit'].fillna(0, inplace=True)
    return H


# Bar chart of the feature importances
def feature_import(X_train, y_train, obj, title):
    print 'feature_importance happening...'
    rf = obj
    rf.fit(X_train, y_train)
    importances = rf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in rf.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]


    # Print the feature ranking
    print("Feature ranking:")

    for f in range(X_train.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances, {}".format(title))
    plt.bar(range(15), importances[indices[:15]], color="r", yerr=std[indices[:15]], align="center")
    plt.xticks(range(15), indices[:15])
    plt.xlim([-1, 15])
    # plt.show()
    plt.savefig('feature_importance_{}.png'.format(title))



# Make a plot of the predictions versus the actual for the training set
def pred_plot(pred):
    print "\nprofit_plot...\n"
    X = pd.read_csv('/Users/brenthowe/datascience/data sets/uptake/training.csv')
    actual = X.pop('profit')
    actual.fillna(0, inplace=True)

    l = len(actual)
    print len(actual)
    print len(pred)

    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot(2,1,1)
    ax.scatter(range(l), pred, color ='r', alpha=.6, label='Predicted')
    ax.scatter(range(l), actual, color ='b', alpha=.6, label='Actual')
    plt.title("Predictions versus Actual, Full Dataset")
    plt.legend()
    # plt.show()
    # plt.savefig('pred_v_actual_full.png')

    n = 100
    ind = np.random.choice(l, n, replace=False)

    ax = fig.add_subplot(2,1,2)
    ax.scatter(range(n), pred[ind], color ='r', alpha=.6, label='Predicted')
    ax.scatter(range(n), actual.values[ind], color ='b', alpha=.6, label='Actual')
    plt.title("Predictions versus Actual, Random Subsample")
    plt.legend()
    # plt.show()
    plt.savefig('pred_v_actual_randsub.png')


# Calculate profit per marketing unit
# I'm assuming that in this training dataset these observations represent the outcomes of a marketing campaign, where all of these individuals were contacted.
def rel_pred(pred):
    X = pd.read_csv('/Users/brenthowe/datascience/data sets/uptake/training.csv')
    marketing_units = 8238
    marketing_cost = marketing_units*30.
    profit = X['profit']
    profit.fillna(0, inplace=True)
    profitt = profit.values

    policy_profit = np.sum(profitt)
    total_profit = policy_profit - marketing_cost
    print "Profit per marketing unit, historical data: {}".format(total_profit/float(marketing_units))

    print marketing_units
    print marketing_cost
    print policy_profit
    print total_profit


    marketing_units = np.sum(np.where(pred>30, 1, 0))
    marketing_cost = marketing_units*30
    policy_profit = np.sum(np.where(pred>30, pred, 0))
    total_profit = policy_profit - marketing_cost
    print "Profit per marketing unit, predicted: {}".format(total_profit/float(marketing_units))


# write the predictions to file
def to_file(offer_vec):
    holdout = pd.read_csv('/Users/brenthowe/datascience/data sets/uptake/testingCandidate.csv')
    holdout['offer'] = offer_vec
    holdout.to_csv('/Users/brenthowe/datascience/data sets/uptake/testingCandidate_predict.csv')



if __name__=="__main__":
    #load and process data
    X_c_train, y_c_train, X_r_train, y_r_train = load_data('/Users/brenthowe/datascience/data sets/uptake/training.csv')
    # train two predictors...classifier for the purchase step and regressor for the profit predict step
    rf_c_obj = respond_classifier(X_c_train, y_c_train)
    rf_r_obj = profit_regress(X_r_train, y_r_train)

    #load testing data
    X_test = load_data('/Users/brenthowe/datascience/data sets/uptake/testingCandidate.csv', train=False)
    # predict using testing data
    offer_vec1 = profit_predict(X_test, rf_c_obj, rf_r_obj)

    # write predictions to file
    to_file(offer_vec1['offer'].values)

    offer_vec2 = profit_predict(X_c_train, rf_c_obj, rf_r_obj)
    # make a plot of the predictions versus actual for training data
    pred_plot(offer_vec2['profit'].values)

    # feature importance from both prediction steps
    rf = RandomForestClassifier(max_features=5, n_estimators=100)
    feature_import(X_c_train, y_c_train, rf, 'Responded')
    rf = RandomForestRegressor(max_features=25, n_estimators=100)
    feature_import(X_c_train, y_c_train, rf, 'Profit')

    # calculation of effect on profit per marketing unit
    rel_pred(offer_vec2['profit'].values)

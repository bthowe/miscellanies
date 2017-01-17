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

        print X_c_enc.info()
        print X_r_enc.info()

        print y_c.describe()
        print y_r.describe()

        return X_c_enc, y_c, X_r_enc, y_r
    else:
        X_c = df
        enc = OneHotEncoder(categorical_features = [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 19])
        X_c_enc = enc.fit_transform(X_c).toarray()
        return X_c_enc




    # if train:
    #     #### responded
    #     le.fit(df['responded'])
    #     df['responded'] = le.transform(df['responded'])
    #
    #     #### profit....looks fine
    #
    #     #### dataset conditional on purchase
    #     df_profit = df[df.profit.notnull()]
    #     df_profit.drop(['responded'], 1, inplace=True)
    #     y_r = df_profit.pop('profit')
    #     X_r = df_profit
    #     enc = OneHotEncoder(categorical_features = [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 19])
    #     X_r_enc = enc.fit_transform(X_r).toarray()
    #
    #     #### unconditional dataset
    #     df.drop(['profit'], 1, inplace=True)
    #
    #     y_c = df.pop('responded')
    #     X_c = df
    #
    #     enc = OneHotEncoder(categorical_features = [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 19])
    #     X_c_enc = enc.fit_transform(X_c).toarray()
    #
    #     return X_c_enc, y_c, X_r_enc, y_r
    # else:
    #     X_c = df
    #     enc = OneHotEncoder(categorical_features = [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 19])
    #     X_c_enc = enc.fit_transform(X_c).toarray()
    #     return X_c_enc



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



def profit_predict(X, c_obj, r_obj):
    print "\nprofit_predict...\n"
    df = pd.DataFrame(X)

    ind = np.where(c_obj.predict(X)==1)

    offer = np.where(r_obj.predict(df.iloc[ind])>30, 1, 0)

    a = np.array([ind[0], offer]).T

    ind_offer = pd.DataFrame(a, columns = ['ind', 'offer'])

    offer_indices = ind_offer[ind_offer.offer==1].ind.values

    temp = pd.DataFrame(np.zeros(len(X_test)))

    temp.iloc[offer_indices]=1

    return temp.values



def to_file(offer_vec):
    holdout = pd.read_csv('/Users/brenthowe/datascience/data sets/uptake/testingCandidate.csv')
    holdout['offer'] = offer_vec
    holdout.to_csv('/Users/brenthowe/datascience/data sets/uptake/testingCandidate_predict.csv')



if __name__=="__main__":
    X_c_train, y_c_train, X_r_train, y_r_train = load_data('/Users/brenthowe/datascience/data sets/uptake/training.csv')
    rf_c_obj = respond_classifier(X_c_train, y_c_train)
    rf_r_obj = profit_regress(X_r_train, y_r_train)

    X_test = load_data('/Users/brenthowe/datascience/data sets/uptake/testingCandidate.csv', train=False)
    # print pd.DataFrame(X_test).info() # same number of variables...looking specifically at the encoded variables, it looks like the encoding was the same.
    offer_vec = profit_predict(X_test, rf_c_obj, rf_r_obj)

    to_file(offer_vec)




# Something that could inhibit generalization(?) to other datasets is the encoding step...wouldn't be able to use the model I train above necesssarily




    #### How do I handle the days since contact variable? If I do a tree based approach, create a variable called
        # maybe I'll just bin it.

    #### I can see a potential correlation between the financial indicators and a purchase
    #### what is the relationship between number of employees (I'm assuming this is at the insurance company) and a purchase?




    #### responded...about 11% purchased in training set



# presentation to non-technical client

    # which variables seem to be the most significant in predicting whether they purchase and whether they are profitable?
    # what type of increase in profit are we talking about here?
        # compare the predicted profit with the historical


        # diagram of predictions...what I predict and what the data actually are
        # which features are most important in the two separate stages
        # 

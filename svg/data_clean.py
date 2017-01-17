import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder



def zip_average_income():
    df = pd.read_csv('/Users/brenthowe/datascience/galvanize/project/data/14zpallagi.csv')
    df_sum = df.groupby('zipcode')[['N02650', 'A02650']].sum()
    df_sum['mean_income'] = df_sum['A02650']/df_sum['N02650']
    df_sum.drop(['N02650', 'A02650'], 1, inplace=True)
    df_sum['zip'] = df_sum.index.astype('str')
    df_sum['zip'] = df_sum.zip.apply(lambda x: x.zfill(5))
    df_sum.set_index('zip', inplace=True)

    us_income = float(df_sum.loc['00000'].values)
    df_sum.drop(df_sum.index[0], inplace=True)
    df_sum['zip'] = df_sum.index
    return us_income, df_sum



def data_clean(path, train=True):
    print 'data_clean is executing'
    df = pd.read_csv(path)
    le = LabelEncoder()

    #### Unnamed: 0 and System ID
    # print pd.DataFrame(df['Unnamed: 0'].unique()).describe() # This looks like a unique identifier
    # print pd.DataFrame(df['System ID'].unique()).describe() # 137042 unique values here...likely doesn't carry any useful information
    df.drop(['Unnamed: 0', 'System ID'], 1, inplace=True)

    #### Created Date Time
    df['Created Date Time'] = pd.to_datetime(df['Created Date Time']) # convert to pandas datetime object
    df['day_of_week'] = df['Created Date Time'].dt.dayofweek # create day of week variable
    df['month'] = df['Created Date Time'].dt.month # create month variable
    # no major holidays in dataset time frame so won't create a holiday dummy
    df.drop(['Created Date Time'], 1, inplace=True)

    #### Neustar Result Code
    # print pd.DataFrame(df['Neustar Result Code'].unique()).describe() #6 unique values
    # print df['Neustar Result Code'].unique()
    df['Neustar Result Code'] = df['Neustar Result Code']
    df['Neustar Result Code'].fillna(value=999, inplace=True)

    #### Lead Source
    # print pd.DataFrame(df['Lead Source'].unique()).describe() #183 unique values
    le.fit(df['Lead Source'])
    df['Lead Source'] = le.transform(df['Lead Source'])

    #### Smoker
    # print df['Smoker'].unique()
    df.ix[df['Smoker']=='FALSE', 'Smoker'] = 'No'
    df.ix[df['Smoker']=='OE State (Not Required)', 'Smoker'] = np.NaN
    df.ix[df['Smoker']=='N', 'Smoker'] = 'No'
    df.ix[df['Smoker']=='1', 'Smoker'] = 'Yes'
    df.ix[df['Smoker']=='TZT.Leads.Runtime.Domain.Models.Field', 'Smoker'] = np.NaN
    df.ix[df['Smoker']=='TRUE', 'Smoker'] = 'Yes'
    df.ix[df['Smoker']=='Y', 'Smoker'] = 'Yes'
    df['Smoker'].fillna(value='0', inplace=True)
    le.fit(df['Smoker'])
    df['Smoker'] = le.transform(df['Smoker'])

    #### Emails
    # print df['Emails'].unique()
    df['Emails'].fillna(value=df['Emails'].median(), inplace=True) #use median because the distribution is skewed
    le.fit(df['Emails'])
    df['Emails'] = le.transform(df['Emails'])

    #### Birthdate
    # Looks like birthdate may be age in days, so I'll leave it as it is.
    # Use median value in place of a missing value
    df['Birthdate'].fillna(value=df['Birthdate'].median(), inplace=True) #use median because the distribution is skewed

    #### Gender
    df.ix[df['Gender']=='F', 'Gender'] = 'Female'
    df.ix[df['Gender']=='M', 'Gender'] = 'Male'
    df['Gender'].fillna(value='0', inplace=True)
    le.fit(df['Gender'])
    df['Gender'] = le.transform(df['Gender'])

    #### Applicant State/Province
    # print df['Applicant State/Province'].unique()
    df['Applicant State/Province'].fillna(value='0', inplace=True)
    le.fit(df['Applicant State/Province'])
    df['Applicant State/Province'] = le.transform(df['Applicant State/Province'])

    #### Applicant Zip/Postal Code
    df['l'] = df['Applicant Zip/Postal Code'].astype(str).apply(lambda x: len(x))
    df.ix[(df['l']>5) & (df['l']<10), 'Applicant Zip/Postal Code'] = np.NaN
    df['Applicant Zip/Postal Code'] = df['Applicant Zip/Postal Code'].astype(str).apply(lambda x: x.split('-')[0] if len(x)>5 else x)
    df['Applicant Zip/Postal Code'] = df['Applicant Zip/Postal Code'].astype(str).apply(lambda x: '0' + x if len(x)==4 else x)
    df['Applicant Zip/Postal Code'] = df['Applicant Zip/Postal Code'].astype(str).apply(lambda x: '00' + x if (len(x)==3) & (x!='nan') else x)
    df['zip'] = df['Applicant Zip/Postal Code']
    df.drop(['l', 'Applicant Zip/Postal Code'], 1, inplace=True)

    #### mean_income
    us_income, df_sum = zip_average_income()
    df_sum['mean_income'] = df_sum['mean_income']*1000
    df = df.merge(df_sum, how='left', on='zip')
    df['mean_income'].fillna(value=df['mean_income'].mean(), inplace=True) #approximate mean income in the United States

    #### median_income
    df_median = pd.read_csv('income_by_zipcode.txt', sep='\t', index_col=False)
    df_median['zip'] = df_median['zipcode, median_income'].str[:5]
    df_median['zip'] = df_median['zip'].apply(lambda x: '0'+ x if len(x)==4 else x)
    df_median['median_income'] = df_median['zipcode, median_income'].str[8:].apply(lambda x: x.replace(",",""))
    df_median.drop('zipcode, median_income', 1, inplace=True)
    df = df.merge(df_median, how='left', on='zip')
    df.ix[(df.median_income == '') | (df.median_income == ' '), 'median_income'] = np.nan
    df.median_income = df.median_income.astype(float)
    df['median_income'].fillna(value='51939', inplace=True) #approximate median income in the United States
    df['median_income'] = df['median_income'].astype(float)
    df.drop(['zip', 'Applicant City'], 1, inplace=True)

    #### target
    if train:
        df['target'].fillna(value=0, inplace=True)
        X = df[['Neustar Result Code', 'Lead Source', 'Smoker', 'Emails', 'Birthdate', 'Gender', 'Applicant State/Province', 'day_of_week', 'month', 'mean_income', 'median_income']]
        y = df.pop('target')
        enc = OneHotEncoder(categorical_features = [0, 1, 2, 5, 6, 7, 8])
        X_enc = enc.fit_transform(X).toarray()
        return X_enc, y

    else:
        X = df[['Neustar Result Code', 'Lead Source', 'Smoker', 'Emails', 'Birthdate', 'Gender', 'Applicant State/Province', 'day_of_week', 'month', 'mean_income', 'median_income']]
        enc = OneHotEncoder(categorical_features = [0, 1, 2, 5, 6, 7, 8])
        X_enc = enc.fit_transform(X).toarray()
        return X_enc, y

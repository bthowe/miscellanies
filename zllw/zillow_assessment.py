import pandas as pd
import numpy as np
from numpy.random import uniform
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

def first_time():
## I first read the data in and then pickled it since reading it in every time I ran this script takes a long time
    df = pd.read_csv('ListingsAndSales.csv') #this file must be in the current directory
    df.to_pickle('listings_and_sales') #saves this file to the current directory

def question1():
    df = pd.read_pickle('listings_and_sales') #read in the data...file should be in the current directory
    df_sales = df[['ListingDate', 'SalesDate', 'Latitude', 'Longitude']] #only want to keep these four variables
    df_sales.ListingDate = pd.to_datetime(df_sales.ListingDate) #covert this to time variable
    df_sales.SalesDate = pd.to_datetime(df_sales.SalesDate) #covert this to time variable

## create time to sale variable...equal to days between initial listing and sale
    df_sales['time_to_sale'] = (df_sales.SalesDate - df_sales.ListingDate).dt.days

##bin the days to sale variable into one of five categories: 0 if the home does not sell, 1 if it sells and takes more than 37 days, etc.
    df_sales['tts'] = 0
    df_sales.ix[df_sales.time_to_sale > 37, 'tts'] = 1
    df_sales.ix[df_sales.time_to_sale <= 37, 'tts'] = 2
    df_sales.ix[df_sales.time_to_sale <= 30, 'tts'] = 3
    df_sales.ix[df_sales.time_to_sale <= 22, 'tts'] = 4
    time_to_sale = df_sales.tts.values
## how did i come up with these threshold values? They are the 25, 50, and 75 perctiles of the distribution of days between listing and sale

## now I get into plotting the categorical days to sale variable I just created on a map of Seattle
    np_lat = map(str, df_sales['Latitude'].values)
    latitude = map(float, np.array(["{0}.{1}".format(lat[:2], lat[2:]) for lat in np_lat]))

    np_long = map(str, df_sales['Longitude'].values)
    longitude = map(float, np.array(["{0}.{1}".format(lon[:4], lon[4:]) for lon in np_long]))

## I define the ranges of longitude and latitude for the map
    long1 = -122.4417888
    long2 = -122.194833
    lat1 = 47.472277
    lat2 = 47.752659

    m = Basemap(projection='gall',
                  llcrnrlon = long1, #-15,              # lower-left corner longitude
                  llcrnrlat = lat1,  #28,               # lower-left corner latitude
                  urcrnrlon = long2, #45,               # upper-right corner longitude
                  urcrnrlat = lat2, #73,               # upper-right corner latitude
                  resolution = 'h', #None, 'c', 'l', 'i', 'h', 'f'
                  area_thresh = 100000.0,
                  )

    bins = 30 #size of the hexagons
    x1, y1 = m(np.array(longitude), np.array(latitude)) #covert data to map points
    z = np.array(time_to_sale) #the outcome of interest which defines the color on the map
    fig = plt.figure(figsize=(12,5))
    ax = fig.add_subplot(1,1,1)
    CS = m.hexbin(x1, y1, C = z, gridsize=bins, cmap=plt.cm.jet)
    m.drawcoastlines()
    m.fillcontinents(color = 'gainsboro', zorder=0)
    m.drawmapboundary(fill_color='steelblue')
    m.colorbar()
    plt.title('Speed of Sale by Location')
    plt.show()

def question2():
    df = pd.read_pickle('listings_and_sales') #read in the data

## check whether there are some observations such that SalesDate is missing but SaleDollarCnt is not
    # print df[pd.isnull(df.SalesDate) & ~pd.isnull(df.SaleDollarCnt)]

## check whether there are some observations such that SaleDollarCnt is missing but SalesDate is not
    # print df[~pd.isnull(df.SalesDate) & pd.isnull(df.SaleDollarCnt)]
    ## there are four such observations...drop these
    df = df[~(~pd.isnull(df.SalesDate) & pd.isnull(df.SaleDollarCnt))]

## assume for all observations such that SaleDollarCnt and SalesDate are missing that the house has not been sold
## create a dummy equal to 1 if the house sells before April 7th, and 0 otherwise
    df['sold'] = 1
    df.ix[pd.isnull(df.SalesDate), 'sold'] = 0
    # print df.sold.describe() #...looks like 32% of homes sold listed before April 7th, sold before April 7th.

    df.ListingDate = pd.to_datetime(df.ListingDate) #transform listingdate into a pandas time variable

## distribution of listings by day
    # fig = plt.figure(figsize=(12,5))
    # ax = fig.add_subplot(1,1,1)
    # ax.bar(df.ListingDate.unique() , df.groupby('ListingDate')['ListingDate'].count())
    # plt.show()
    ## looks mostly uniform...maybe a greater concentration of listings come post February

    df.SalesDate = pd.to_datetime(df.SalesDate)
    df['time_to_sale'] = (df.SalesDate - df.ListingDate).dt.days #create variable number of days to sale
    df['listing_month'] = df.ListingDate.dt.month #create binary variable month of listing
    df['days_since_listing'] = (pd.to_datetime('2015-04-07') - df.ListingDate).dt.days #create a variable called days since listing I use in the analysis

## conditional on selling, what is the distribution of sale times.
    # dftts = df.time_to_sale
    # dftts.dropna(inplace=True)
    # print dftts.describe()
    #
    # fig = plt.figure(figsize=(12,5))
    # ax = fig.add_subplot(1,1,1)
    # ax.hist(dftts)
    # plt.show()

## since there are so many nulls values in MajorRemodelYear, I create a dummy equal to 1 if it is nonnull
    df['remodel'] = 1
    df.ix[pd.isnull(df.MajorRemodelYear), 'remodel'] = 0

####################
## the next few lines finesse the data, with the end result being my vector of outcomes and matrix of covariates
    df_time_to_sale = df.pop('time_to_sale')
    ## drop the null values since missing at random...maybe dropping the estimate is a good idea too
    df_data = df.drop(['ListingDate', 'SalesDate', 'SaleDollarCnt', 'MajorRemodelYear', 'RoomTotalCnt', 'Latitude', 'Longitude'], 1)
    # leaving 'ZestimateDollarCnt' in seems to increase performance even though it reduces number of observations
    df_data.dropna(inplace = True) #drop missing values

    index = df_data.index.values #get the indeces of the values I keep to keep the correct observations in the outcomes vector, next line
    df_data['time_to_sale'] = df_time_to_sale[index]
    y_class = df_data.pop('sold')

## I define the training and test sets for the two models I fit below
    X_train, X_test, y_class_train, y_class_test = train_test_split(df_data, y_class, test_size=0.3, random_state = 23098)
    y_reg_train = X_train.pop('time_to_sale')
    y_reg_test = X_test.pop('time_to_sale')

## this defines data I use in the plot of actual versus predicted below
    y_test = y_reg_test.copy() #actual time to sale values
    y_test.ix[pd.isnull(y_test)]=0 #set the missing values to zero (i.e., if no sale occurred)

## classification model: binary variable is the outcome whether the property sold or not
    gbc = GradientBoostingClassifier(learning_rate = 1.425)
    gbc.fit(X_train, y_class_train)
    y_class_pred = gbc.predict(X_test)

    accuracy = accuracy_score(y_class_pred, y_class_test) #measure of how well the model does

## regresssion: conditional on selling, time to sale in days is the outcome variable
    y_reg_train.dropna(inplace=True) #drop missing values
    y_reg_test.dropna(inplace=True)
    index_train = y_reg_train.index.values #again, get indeces so I can make sure I can only retain the observations not dropped in the previous two lines
    index_test = y_reg_test.index.values

    X_reg_train = X_train.loc[y_reg_train.index.values, :]
    X_reg_test = X_test.loc[y_reg_test.index.values, :]

    gbr = GradientBoostingRegressor(learning_rate = 1)
    gbr.fit(X_reg_train, y_reg_train)
    y_reg_pred = gbr.predict(X_reg_test)

    mse = mean_squared_error(y_reg_pred, y_reg_test) #measure of how well the model does

## plot all of the actual data and their predictions
    y_pred = [] #I am generating a list of predicted values for each of the data points
    for obs in X_test.values:
        pred = gbc.predict(obs)
        if pred==1:
            y_pred.append(gbr.predict(obs))
        else:
            y_pred.append(0)

    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot(1,1,1)
    ax.scatter(range(len(y_pred)), y_pred, color ='r', alpha=.6, label='Predicted')
    ax.scatter(range(y_test.shape[0]), y_test, color ='b', alpha=.6, label='Actual')
    plt.legend()
    plt.show()


## print the accuracy score and mse
    print '\n\n\n\n\n'
    print "Test data accuracy (classificaiton): {}".format(accuracy)
    print '\n'
    print "Test data mean-squared-error (regression): {}".format(mse)




## the following defines what is executable from the command line
if __name__=="__main__":
## each of the three functions are executed
    first_time()
    question1()
    question2()

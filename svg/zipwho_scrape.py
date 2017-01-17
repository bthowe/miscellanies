import time
import requests
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

def incomebyzipcode():
    df = pd.read_csv('/Users/brenthowe/datascience/data sets/svg/train_test_.csv')
    with open('zipcode_info.txt', 'a') as f:
        f.write('{0}, {1}\n'.format('zipcode', 'median_income'))
    for zipcode in df['Applicant Zip/Postal Code'].unique():
        time.sleep(np.random.randint(10))
        url = "https://www.incomebyzipcode.com/search/{}".format(zipcode)
        # url = "https://www.incomebyzipcode.com/search/{}".format(80921)
        # url = "http://zipwho.com/?zip={}&mode=zip".format(zipcode)
        # url = "http://zipwho.com/?zip=60626&mode=zip"
        req = requests.get(url)
        sc = req.status_code
        if sc==200:
            soup = BeautifulSoup(req.content, 'html.parser')

            median_income = soup.findAll("td", {"class" : "hilite"})[0].text
            print median_income

        else:
            median_income = ''
            print 'Nope'

        with open('zipcode_info.txt', 'a') as f:
            f.write('{0}, {1}\n'.format(zipcode, median_income))

# I'm still blacklisted when I try and scrape but not when I visit the website through Chrome
def zipwho():
    df = pd.read_csv('/Users/brenthowe/datascience/data sets/svg/train_test_.csv')
    with open('zipcode_info_complete.txt', 'a') as f:
        f.write('{0}, {1}, {2}, {3}, {4}, {5}, {6}\n'.format('zipcode', 'median_income', 'college_degree', 'white', 'black', 'asian', 'hispanic'))
    for zipcode in df['Applicant Zip/Postal Code'].unique():
        print zipcode

        time.sleep(np.random.randint(10))

        url = "http://zipwho.com/?zip={}&mode=zip".format(zipcode)
        # url = "http://zipwho.com/?zip=60626&mode=zip"
        req = requests.get(url)
        sc = req.status_code
        if sc==200:
            soup = BeautifulSoup(req.content, 'html.parser')
            if soup=='?':
                print "blacklisted!"
                break
            print soup
            median_income = soup.script.text.split('"')[1].split(',')[39]
            college_degree = soup.script.text.split('"')[1].split(',')[49]
            white = soup.script.text.split('"')[1].split(',')[65]
            black = soup.script.text.split('"')[1].split(',')[67]
            asian = soup.script.text.split('"')[1].split(',')[69]
            hispanic = soup.script.text.split('"')[1].split(',')[71]
        else:
            median_income = ''
            college_degree = ''
            white = ''
            black = ''
            asian = ''
            hispanic = ''

        with open('zipcode_info_complete.txt', 'a') as f:
            f.write('{0}, {1}, {2}, {3}, {4}, {5}, {6}\n'.format(zipcode, median_income, college_degree, white, black, asian, hispanic))

if __name__=="__main__":
    # incomebyzipcode()
    zipwho()

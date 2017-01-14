import numpy as np
import pandas as pd

def load_data():
    df = pd.read_csv('/Users/brenthowe/datascience/data sets/svg/train_test_.csv')
    print df.info()
    print df.describe()




if __name__=="__main__":
    load_data()

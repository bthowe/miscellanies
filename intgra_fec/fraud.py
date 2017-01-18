# import important packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def number_of_procedures():
    df = pd.read_csv('Downloads/Current_Test_Data.csv', dtype={'provider':pd.np.str, 'physician':pd.np.str}) #read in the data

    df = df[['id_number', 'provider', 'physician', 'procedure1' ,'procedure2' ,'procedure3' ,'p_diag' ,'total_charges']] #choose the columns I want

# deal with missing observations
    df.physician = df.physician.fillna('0')
    df.procedure2 = df.procedure2.fillna('0')
    df.procedure3 = df.procedure3.fillna('0')
    df.dropna(inplace=True) #drop any other observations with a variable missing

    df_PD = df.groupby(['p_diag']).mean().sort('total_charges')
    print df_PD
    # 61882, 80426, V5302, V5332, 99690

#keep only these five diagnoses
    mask = (df['p_diag']=='61882') | (df['p_diag']=='80426') | (df['p_diag']=='V5302') | (df['p_diag']=='V5332') | (df['p_diag']=='99690')
    df_proc = df[mask]

#calculate the number of procedures performed
    df_proc.proc2 = 0
    df_proc.ix[df_proc.procedure2!='0', 'proc2'] = 1
    df_proc.proc3 = 0
    df_proc.ix[df_proc.procedure3!='0', 'proc3'] = 1

    df_proc['proc_num'] = df_proc['proc2'] + df_proc['proc3'] + 1

    print df_proc['proc_num'].describe() #the summary stats for the number of procedures for these five diagnoses pooled


def number_of_days():
    #this code is similar to above...instead of number of procedures I use service days
    df = pd.read_csv('Downloads/Current_Test_Data.csv', dtype={'provider':pd.np.str, 'physician':pd.np.str})

    df = df[['id_number', 'provider', 'physician', 'procedure1' ,'procedure2' ,'procedure3' ,'p_diag' ,'service_days']]

    df.physician = df.physician.fillna('0')
    df.procedure2 = df.procedure2.fillna('0')
    df.procedure3 = df.procedure3.fillna('0')
    df.dropna(inplace=True)

    df_PD = df.groupby(['p_diag']).mean().sort('service_days')
    print df_PD
# 94434 94332 1422 V8404 4847
    mask = (df['p_diag']=='94434') | (df['p_diag']=='94332') | (df['p_diag']==' 1422') | (df['p_diag']==' V8404') | (df['p_diag']=='4847')
    df_sd = df[mask]

    print df_sd.describe() #the summary stats for the number of service days for these five diagnoses pooled...again, the distribution is degenerate so I do something different

    df['p_diag'].unique().shape #this tells me there are 8035 unique p_diags

    codes = np.random.choice(df['p_diag'].unique(), 5, replace=False)
    print codes
    # codes = np.array(['70441' '37160' '8170' '1538' 'V131'])

    # plots the histograms for each of these codes...note, the five codes selected will be different every time this is run since I use a random selection
    fig = plt.figure()
    for ind, code in enumerate(codes):
        mask = df['p_diag']==code
        # df[mask]['service_days'].plot.hist()

        ax = fig.add_subplot(5,1,ind+1)
        ax.hist(df[mask]['service_days'], label = code, normed=1)
        # ax.set_xlim([0, 5])
        plt.legend()
    plt.show()


if __name__=="__main__":
    number_of_procedures() #executes first function
    number_of_days() #executes second function

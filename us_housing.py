
import pandas as pd
from lxml import html
import quandl
import pickle
import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')



def state_list():
    fiddy_states = pd.read_html('https://simple.wikipedia.org/wiki/List_of_U.S._states')
    return fiddy_states[0][1][1:]


def grab_initial_state_data():

    quandl.ApiConfig.api_key = 'NqEjStuNK6jCRByXknos'
    states = state_list()

    main_df = pd.DataFrame()

    for abbv in states:
        query = "FMAC/HPI_"+str(abbv)
        df = quandl.get(query)
        df.rename(columns={'Value':str(abbv)}, inplace=True)
        print(query)
        df[abbv] = (df[abbv] - df[abbv][0]) / df[abbv][0] * 100.0
        print(df.head())
        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df)

    pickle_out = open('fiddy_states3.pickle','wb')
    pickle.dump(main_df, pickle_out)
    pickle_out.close()

def HPI_benchmark():
    quandl.ApiConfig.api_key = 'NqEjStuNK6jCRByXknos'
    df = quandl.get("FMAC/HPI_USA")
    df.rename(columns={'Value':"United States"}, inplace=True)
    df["United States"] = (df["United States"]-df["United States"][0]) / df["United States"][0] * 100.0
    return df

def gdp_data():
    quandl.ApiConfig.api_key = 'NqEjStuNK6jCRByXknos'
    query = "BCB/4385"
    df = quandl.get(query, trim_start="1975-01-01")
    df["Value"] = (df["Value"]-df["Value"][0]) / df["Value"][0] * 100.0
    df=df.resample('M').mean()
    df.rename(columns={'Value':'GDP'}, inplace=True)
    return df

def us_unemployment():
    quandl.ApiConfig.api_key = 'NqEjStuNK6jCRByXknos'
    query = "ECPI/JOB_G"
    df = quandl.get(query, trim_start="1975-01-01")
    df["Unemployment Rate"] = (df["Unemployment Rate"]-df["Unemployment Rate"][0]) / df["Unemployment Rate"][0] * 100.0
    df=df.resample('1D').mean()
    df=df.resample('M').mean()
    df.rename(columns={'Value':'GDP'}, inplace=True)
    return df

def mortgage_30y():
    quandl.ApiConfig.api_key = 'NqEjStuNK6jCRByXknos'
    query = "FMAC/MORTG"
    df = quandl.get(query, trim_start="1975-01-01")
    df["Value"] = (df["Value"]-df["Value"][0]) / df["Value"][0] * 100.0
    df=df.resample('1D').mean()
    df=df.resample('M').mean()
    df.columns = ['M30']
    return df

def sp500_data():
    quandl.ApiConfig.api_key = 'NqEjStuNK6jCRByXknos'
    query = "YAHOO/INDEX_GSPC"
    df = quandl.get(query, trim_start="1975-01-01")
    df["Adjusted Close"] = (df["Adjusted Close"]-df["Adjusted Close"][0]) / df["Adjusted Close"][0] * 100.0
    df=df.resample('M').mean()
    df.rename(columns={'Adjusted Close':'sp500'},inplace=True)
    return df

m30 = mortgage_30y()
#US_unemployment = us_unemployment()
US_GDP = gdp_data()
#sp500 = sp500_data()
HPI_Bench = HPI_benchmark()
HPI_data = pd.read_pickle('fiddy_states3.pickle')

HPI = HPI_data.join([HPI_Bench, m30, US_GDP])
print(HPI.corr())

HPI.to_pickle('HPI.pickle')

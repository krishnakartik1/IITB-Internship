import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor


def demandForecasting(df):
    regressionTree = BaggingRegressor(DecisionTreeRegressor(min_samples_leaf=10),n_estimators=100,max_samples=df.shape[0])
    regressionTree.fit(df.iloc[:,:-1], df.iloc[:,-1])
    return regressionTree

def prediction(regressionTree,df):
    newStylesDemand=regressionTree.predict(df)
    df["demand"]=newStylesDemand
    return df

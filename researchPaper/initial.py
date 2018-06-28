import numpy as np
import pandas as pd
from demandPrediction import demandForecasting, prediction
from salesPrediction import salesClac
from priceOptimization import LPBound

oldStyles=pd.read_csv("oldStyles.csv")
firstExposureStyle=pd.read_csv("firstExposureStyle.csv")

# preprocessing done to get demand of soldOutData

regressionTree=demandForecasting(oldStyles)
firstExposureStyle["demand"]=prediction(regressionTree,firstExposureStyle)

minP=0.0 # minimum price for the set of prices
maxP=1000.0 # maximum price for the set of prices
final=salesClac(firstExposureStyle["demand","type"])

prices=[]
for i in np.unique(firstExposureStyle["type"].value()):
    prices.append(LPBound(final[final["type"]==i], list(range(minP,maxP+1,5)), final[final["type"]==i]["salesP"]))
    

    



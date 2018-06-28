import numpy as np
import pandas as pd

#===============================================================================
# Size curve has percent of style demand that should be allocated to each size.
#  
#===============================================================================
def salesClac(demandp):
    sizeCurve=pd.read_csv("sizeCurve.csv") 
    inventory=pd.read_csv("inventory.csv")
    
    sizeCurveNP=np.array(sizeCurve)
    sizeSplit=[sizeCurveNP[demandp.iloc[i,"type"].value()]*demandp["demand"].value() for i in demandp.shape[0]]
    
    sizeWiseSales=np.minimum(inventory,sizeSplit)

    demandp["salesP"]=np.sum(sizeWiseSales, 1)
    return demandp



from pulp import *
import numpy as np
from audioop import reverse
import operator as op


def optimizeLP(countN,m,d,k):
    
    countM=len(m)
    
    prob=LpProblem("LP relaxation",LpMaximize)
    
    xLabel=[]
    
    for i in range(countM*countN):
        xLabel.append(i)
        
    x=LpVariable.dict("x", xLabel, 0, 1)
    
    prob+=lpSum((m*d)[i/countM,i%countM]*x[i] for i in xLabel)
    
    for i in range(countN):
        prob+=lpSum(x[j] for j in range(i*countM,i*countM+countM))==1
        
    onesConstraint=np.ones(d.shape)
    prob+=lpSum((m*onesConstraint)[i/countM,i%countM]*x[i] for i in xLabel)==k
    
    prob.solve()
    
    return prob


def optimizeIP(countN,m,d,k):
    
    countM=len(m)
    
    prob=LpProblem("LP relaxation",LpMaximize)
    
    xLabel=[]
    
    for i in range(countM*countN):
        xLabel.append(i)
        
    x=LpVariable.dict("x", xLabel, 0, 1,LpInteger)
    
    prob+=lpSum((m*d)[i/countM,i%countM]*x[i] for i in xLabel)
    
    for i in range(countN):
        prob+=lpSum(x[j] for j in range(i*countM,i*countM+countM))==1
        
    onesConstraint=np.ones(d.shape)
    prob+=lpSum((m*onesConstraint)[i/countM,i%countM]*x[i] for i in xLabel)==k
    
    prob.solve()
    
    return prob


def LPBound(countN,m,d):
    d=np.array(d)
    countM=len(m)
    possiblek=[i for i in np.arange(countN*min(m),countN*max(m)+1,5)]
    zLP={}
    lb={}
    for i,j in zip(possiblek,range(len(possiblek))):
        probLP=optimizeLP(countN, m, d[j], i)
        zLP[i]=value(probLP.objective)
        lb[i]=zLP[i]-np.max((np.max(d[j]*m,1)-np.min(d[j]*m, 1)))
        
#     sortedK=[x for _,x in zip(zLP,possiblek)]
    sortedK=sorted(zLP.items(),key=op.itemgetter(1),reverse=True)
    lbMax=max(lb.items(), key=op.itemgetter(1))[1]
    kCap=max(lb.items(), key=op.itemgetter(1))[0]
    
    for i in sortedK:
        probIP=optimizeIP(countN,m,d[i],i)
        zIP=value(probIP.objective)
        if zIP>lbMax:
            kCap=i
            lbMax=zIP
        
        if zLP[i+1]<=lbMax:
            return probIP #prob of type pulp, because you need the variable values 
    
    return probIP
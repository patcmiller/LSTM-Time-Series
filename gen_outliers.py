import pandas as pd
import numpy as np
from time import time

def compareAnomalies(real, pred):
    nAnomalies= np.sum(real)
    pAnomalies= np.sum(pred)
    nCorrect= np.sum(real[(real==pred) & (real==1)])
    ans= np.subtract(real, pred)
    fPositives= (ans < 0).sum()
    fNegatives= (ans > 0).sum()
    return [nAnomalies, pAnomalies, nCorrect, fPositives, fNegatives]

def getOutliers(ts):

    # MEDIAN ABSOLUTE DEVIATION
    wmed= np.nanmedian(ts)
    wmad= 1.483* np.nanmedian(np.abs(ts-wmed))
    
    if wmad==0: return [0 for t in ts]
    tmp= [np.abs(ts[i] - wmed) / float(wmad) for i in range(len(ts))]
    outliers1= [1 if t > 6.0 else 0 for t in tmp]
    
    # MEAN ABSOLUTE DEVIATION
    wmean= np.nanmean(ts)
    wstd= np.nanstd(ts)
    
    if wstd==0: return [0 for t in ts]
    tmp= [np.abs(ts[i]- wmean) / float(wstd) for i in range(len(ts))]
    outliers2= [1 if t > 4.0 else 0 for t in tmp]
    
    # MEDIAN OF MEDIANS ABSOLUTE DEVIATION
    wmed4= [np.nanmedian(np.abs(ts-ts[i])) for i in range(len(ts))]
    wmed5= 1.193* np.nanmedian(wmed4)
    outliers3= [1 if ts[i] > wmed+6.0*wmed5 else 0 for i in range(len(ts))]

    tmp= np.add(np.add(outliers1,outliers2),outliers3)
    outliers= [1 if t == 3 else 0 for t in tmp]

    return outliers
import numpy as np
import pandas as pd
from lifelines.statistics import logrank_test

def CIndex(hazards, events, survtime_all, cuda=1):
    """
    hazards: network output
    events: os event
    survtime_all: os time
    """
    if cuda:
        hazards = hazards.data.cpu().numpy()
        events = events.data.cpu().numpy()
        survtime_all = survtime_all.data.cpu().numpy()
    concord = 0.
    total = 0.
    N_test = events.shape[0]
    events = np.asarray(events, dtype=bool)
    for i in range(N_test):
        if events[i] == 1:
            for j in range(N_test):
                if survtime_all[j] > survtime_all[i]:
                    total = total + 1
                    if hazards[j] < hazards[i]: concord = concord + 1
                    elif hazards[j] == hazards[i]: concord = concord + 0.5

    return(concord/total)

def CoxLogRank(hazards, events, survtime_all, cuda=1):
    """
    hazards: network output
    events: os event: 0 for right censored and 1 for death event
    survtime_all: os time 
    """
    if cuda:
        hazards = hazards.data.cpu().numpy().reshape(-1) # flatten the numpy array to # of samples in the batch
        events = events.data.cpu().numpy().reshape(-1)
        survtime_all = survtime_all.data.cpu().numpy().reshape(-1)
    # dichotomize the hazards
    hazards_median = np.median(hazards) #print('Median:', hazards_median)
    hazards_dichotomize = np.zeros([len(hazards)], dtype=int)
    hazards_dichotomize[hazards > hazards_median] = 1  # set low risk group as 0, high risk group as 1
    
    idx = hazards_dichotomize == 0
    T1 = survtime_all[idx] # low risk group
    T2 = survtime_all[~idx] # high risk group
    E1 = events[idx]
    E2 = events[~idx]
    results = logrank_test(T1, T2, event_observed_A=E1, event_observed_B=E2)
    return results.p_value
     
def formatTable(eva, outpath):
    df = pd.DataFrame(eva, index=[0])
    df.to_csv(outpath, sep=',', header=True, index=False)
    return df

def formatCombinedTable(eva, outFile='metrics.csv'):
    pass

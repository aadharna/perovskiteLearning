#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

import pickle

import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')


from dml import ANMM, DMLMJ, NCA, NCMC, LDA, NCMML, MCML

from joblib import dump, load

from dml import tune_knn


# In[3]:


# DATAFILE = '/home/jupyter/sd2e-community/versioned-dataframes/perovskite/perovskitedata/0057.perovskitedata.csv'


def _stratify(data0, out, inchis, sampleCutoff):

    stratifiedData0 = pd.DataFrame()
    stratifiedOut = pd.DataFrame()
    
    indicies = {}
    for i, x in enumerate(np.unique(inchis.values.flatten())):
        z = (inchis.values == x).flatten()
        # print(x, z.sum())
        if z.sum() < sampleCutoff:
            continue
        total_amine0 = data0[z].reset_index(drop=True)

        amine_out = out[z].reset_index(drop=True)

        # this is still experimental and can easily be changed.
        uniformSamples = np.random.choice(total_amine0.index, size=sampleCutoff, replace=False)
        sampled_amine0 = total_amine0.loc[uniformSamples]
        
        sampled_out = amine_out.loc[uniformSamples]

        # save pointer to where this amine lives in the stratified dataset.
        # this isn't needed for random-TTS, but makes doing the Leave-One-Amine-Out 
        # train-test-splitting VERY EASY. 
        indicies[x] = np.array(range(96)) + i*96

        stratifiedData0 = pd.concat([stratifiedData0, sampled_amine0]).reset_index(drop=True)
        stratifiedOut = pd.concat([stratifiedOut, sampled_out]).reset_index(drop=True)
    
    stratifiedOut = np.array(stratifiedOut, dtype=int)
    return stratifiedData0, stratifiedOut.squeeze(), indicies


# In[6]:


def _prepare(minimal=False, sampleCutoff=95):
    
    perov = pd.read_csv(DATAFILE, skiprows=4, low_memory=False)
    perov = perov[perov['_raw_expver'] == 1.1].reset_index(drop=True)       
    perov = perov[perov['_raw_reagent_0_chemicals_0_inchikey'] 
                  == "YEJRWHAVMIAJKC-UHFFFAOYSA-N"].reset_index(drop=True)
    # removes three reactions
    perov = perov[perov['_rxn_organic-inchikey'] != 'JMXLWMIFDJCGBV-UHFFFAOYSA-N'].reset_index(drop=True)    
    
    newInchis = perov['_rxn_organic-inchikey'].dropna()
    perov = perov.iloc[newInchis.index].reset_index(drop=True)
    
    inchis = pd.DataFrame.from_dict({"inchis":perov['_rxn_organic-inchikey'].values})
        
    cleanPerov = perov.drop(labels=[raw for raw in perov.columns if "raw" in raw], axis=1)
    cleanPerov = cleanPerov.select_dtypes(exclude=['object'])
    
    cleanPerov.fillna(0, inplace=True)
    cleanPerov['_out_crystalscore'] = np.where(cleanPerov['_out_crystalscore'] == 4, True, False) 
    out = cleanPerov['_out_crystalscore']
    cleanPerov.drop(["_out_crystalscore", 'dataset'], axis=1, inplace=True)
        
    return  _stratify(cleanPerov, out, inchis, sampleCutoff)


# In[7]:

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--alg", type=str, help='One of ANMM, DMLMJ, NCA, NCMC')
parser.add_argument("--combined", default=False, type=bool, help='crystalScore 4 or 3-and-4')
parser.add_argument("--n", default=5, type=int, help='how many neighbors should be used')

args = parser.parse_args()


if __name__ == "__main__":
    
    if args.alg == "ANMM":
        alg = ANMM
        dml_params={}
        tune_args={"num_dims":[2, 3, 5, None],
                   "n_friends":[1, 2, 5],
                   "n_enemies":[1, 2, 5]}
        
    elif args.alg == "DMLMJ":
        alg = DMLMJ
        dml_params={}
        tune_args={"num_dims":[2, 3, 5, None],
                   "n_neighbors":[1, 3, 5]}
        
    elif args.alg == "NCA":
        alg = NCA
        dml_params={}
        tune_args={"num_dims":[2, 3, 5, None],
                   "eta0":[0.1, 0.01]}
        
    elif args.alg == "NCMC":
        alg = NCMC
        dml_params={}
        tune_args={"num_dims":[2, 3, 5, None],
                    "centroids_num":[2, 3, 5],
                    "initial_transform":['euclidean', 'scale']}
        
    elif args.alg == "NCMML":
        alg = NCMML
        dml_params={'learning_rate':"adaptive", 
                    'eta0':1e-6}
        
        tune_args={'num_dims':[2, 3, 5, None], 
                   'initial_transform':['euclidean', 'scale']}
    
    elif args.alg == 'MCML':
        alg = MCML
        dml_params={'learning_rate':"adaptive", 
                    'eta0':1e-6}
        
        tune_args={'num_dims':[2, 3, 5, None], 
                   'initial_metric':['euclidean', 'scale']}
        
    else:
        raise ValueError("alg not supported")
        
################################### END ARGUMENT SETTING ###########################################
###################################
################################### START PREPROCESSING ############################################
    

    perov = pd.read_csv('minimal57Perov.csv')
    perov = perov[perov['_raw_RelativeHumidity'] != -1].reset_index(drop=True)
    inchis = pd.DataFrame.from_dict({"inchis":perov['_rxn_organic-inchikey'].values})
    perov.fillna(0, inplace=True)
    
    if args.combined:
    
        perov['_out_crystalscore'] = np.where(perov['_out_crystalscore'] == 4, True, False) + \
                                     np.where(perov['_out_crystalscore'] == 3, True, False)  
    else:
        perov['_out_crystalscore'] = np.where(perov['_out_crystalscore'] == 4, True, False)
    
    out = perov['_out_crystalscore']
    perov = perov.select_dtypes(exclude=['object'])
    perov.drop(["_out_crystalscore"], axis=1, inplace=True)

    stratPerov, stratOut, indicies = _stratify(perov, out, inchis, 95)

    
    print(f"running CV on {str(alg)}")


    results, best, best, detailed = tune_knn(alg,
                                            X=stratPerov.values, 
                                            y=stratOut,
                                            n_neighbors=args.n,
                                            dml_params=dml_params,
                                            tune_args=tune_args,
                                            n_folds=5,n_reps=2,seed=28,verbose=True)


    # In[ ]:

    combined = "combined" if args.combined else ""
    
    f = open(f"./distanceResults/{args.alg}_{combined}.pkl","wb+")
    pickle.dump(detailed, f)
    f.close()

    dump(best, f"./distanceResults/{args.alg}_{combined}_best.joblib")






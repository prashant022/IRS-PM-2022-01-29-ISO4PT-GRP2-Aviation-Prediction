# Title : For a given Passenger Name and Airline this program recommends the airline rating
# using collaborative filtering.

import pandas as pd
import numpy as np
from random import sample 
from scipy.sparse import issparse
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances


from Utility import mapdata
from Utility import getRecommendations_UU
from Utility import getRecommendations_II
from Utility import sparsity
from Utility import predictRatings
from Utility import getitemsimsmatrix
from Utility import pearsonsim
from Utility import cosinesim
from Utility import euclidsim
from Utility import wtavg
from Utility import makeRatingsMatrix


def getRecommendations_UU(targetrats, ratsmatrix, imap, simfun=pearsonsim, topN=5):

    # get similarity between target and all other users
    sims = []
    for row in ratsmatrix:
      sims.append(simfun(row,targetrats))
    sims = np.array(sims)
    sims[sims < 0] = np.nan

    # for each unseen item, get weighted average of all user ratings
    rats = []
    unseenitemidxs = np.where(np.isnan(targetrats)==True)[0]
    for col in unseenitemidxs:
      rats.append(wtavg(ratsmatrix[:,col], sims))

    # put results into a dataframe and reverse sort by predicted rating
    itemnames=list(imap.keys())
    rats = pd.DataFrame(rats, index=[itemnames[i] for i in unseenitemidxs], columns=['predrating'])
    rats = rats.sort_values(ascending = False, by=['predrating'])
    return rats[0:min(topN,len(rats))]


def pearsonsim(x,y):
    xy = x*y
    x = x[np.isnan(xy)==False]
    y = y[np.isnan(xy)==False]
    if(len(x)==0): return np.nan
    mx=np.mean(x)
    my=np.mean(y)
    rt = np.sqrt(sum((x-mx)**2)*sum((y-my)**2))
    if (rt == 0): return np.nan 
    return sum((x-mx)*(y-my))/rt

def cosinesim(x,y):
    xy = x*y
    x = x[np.isnan(xy)==False]
    y = y[np.isnan(xy)==False]
    if(len(x)==0): return np.nan
    rt = np.sqrt(sum(x**2)*sum(y**2))
    if (rt == 0): return np.nan 
    return sum(x*y)/rt

def euclidsim(x,y):
    xy = x*y
    x = x[np.isnan(xy)==False]
    y = y[np.isnan(xy)==False]
    z=(y-x)**2
    #sz=np.sqrt(sum(z)) 
    sz=sum(z)
    return 1/(1+sz)



def compute_recommend(passenger_name):
    ratings_df = pd.read_csv("Data/aviation_ratings.csv")
    ratings_df.columns = ['user_id','item_id','rating']
    print(ratings_df.shape)
    ratings_df[0:15]
    #Map Data
    ratings_df["item_id"] = ratings_df["item_id"].astype(str)
    ratings_df["user_id"] = ratings_df["user_id"].astype(str)
    ratings_df["rating"]  = ratings_df["rating"].values.astype(np.float32)
    user_ids = np.sort(ratings_df["user_id"].unique()).tolist()
    umap = {x: i for i, x in enumerate(user_ids)}
    item_ids = np.sort(ratings_df["item_id"].unique()).tolist()
    imap = {x: i for i, x in enumerate(item_ids)}
    ratings_df["user_id"] = ratings_df["user_id"].map(umap) # swap userid for user index
    ratings_df["item_id"] = ratings_df["item_id"].map(imap) # swap itemid for item index
    #Make Rating Matrix 
    ratmatrix = pd.pivot_table(ratings_df, index=['user_id'], columns=['item_id'], values=['rating'],aggfunc=[np.mean]).values
    #Assign the target Name
    targetname = passenger_name
    targetrats = ratmatrix[umap[targetname],]
    recommnd = getRecommendations_UU(targetrats, ratmatrix, imap, simfun=pearsonsim)
    return recommnd


def get_airline_ratings(airline_name,passenger_name):
    compute_recom = compute_recommend(passenger_name)
    original_csv = pd.read_csv("Data/aviation_ratings.csv")
    original_csv.head()
    existing_rate = ""
    message = ""
    predict_rate = ""
    for m in original_csv.index:
        if original_csv.Airline[m]== airline_name and original_csv.Passenger[m]==passenger_name:
            existing_rate = str(original_csv.Rating[m])
        else:
            continue
        
    if existing_rate:
        message = "You have rated "+airline_name+" with ratings = "+existing_rate
        return message
    
   
    for i in compute_recom.index:
        if i == airline_name:
            predict_rate = str(compute_recom.predrating[i])
            message = "You may rate "+airline_name+" with ratings = "+predict_rate
            return message
    
    message = "Hmmm ,You yet to rate "+airline_name+"!!"
    return message     




import pandas as pd
import numpy as np
from random import sample 
from scipy.sparse import issparse
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances

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

def jaccardsim(x,y):
    sx = np.where(np.isnan(x)==False)
    sy = np.where(np.isnan(y)==False)
    return float(len(np.intersect1d(sx,sy)))/len(np.union1d(sx,sy))

# weighted average function 
def wtavg(vals, weights):
    xy = vals * weights
    weights = weights[np.isnan(xy) == False] 
    if sum(weights) == 0 : return np.nan
    vals = vals[np.isnan(xy)==False]
    return sum(vals * weights)/sum(weights)

# find similarities between columns in the ratings matrix
# because the matrix is symmetric, we compute the top left half and then copy to bottom right
def getitemsimsmatrix(ratsmatrix, simfun=cosinesim):
    r,c = ratsmatrix.shape
    matrx = list()
    for col1 in range(0,c):
        simrow = [0]*col1
        for col2 in range(col1,c):
            simrow.append(simfun(ratsmatrix[:,col1],ratsmatrix[:,col2]))
        matrx.append(simrow)
    matrx = np.array(matrx)
    matrx = matrx + matrx.T - np.diag(np.diag(matrx))
    return matrx

# map the userids and itemids given in the rating events to contiguous integer indexes
def mapdata(ratings_df):
  ratings_df["item_id"] = ratings_df["item_id"].astype(str)
  ratings_df["user_id"] = ratings_df["user_id"].astype(str)
  ratings_df["rating"]  = ratings_df["rating"].values.astype(np.float32)
  user_ids = np.sort(ratings_df["user_id"].unique()).tolist()
  umap = {x: i for i, x in enumerate(user_ids)}
  item_ids = np.sort(ratings_df["item_id"].unique()).tolist()
  imap = {x: i for i, x in enumerate(item_ids)}
  ratings_df["user_id"] = ratings_df["user_id"].map(umap) # swap userid for user index
  ratings_df["item_id"] = ratings_df["item_id"].map(imap) # swap itemid for item index
  return ratings_df, umap, imap

# create a dense ratings matrix
def makeRatingsMatrix(ratings_df):
  ratings_df, umap, imap = mapdata(ratings_df)
  ratmatrix = pd.pivot_table(ratings_df, index=['user_id'], columns=['item_id'], values=['rating'],aggfunc=[np.mean]).values
  return ratmatrix, umap, imap

# create a sparse ratings matrix
def makeRatingsMatrix_sparse(ratings_df, testsize=0):
  ratings_df, umap, imap = mapdata(ratings_df)
  testidx  = sample(range(ratings_df.shape[0]), testsize)
  testevents = ratings_df.iloc[testidx,].values
  if testsize > 0:
    trainidx = list(set(range(ratings_df.shape[0])) - set(testidx))
    trainevents = ratings_df.iloc[trainidx,]
  else:
    trainevents = ratings_df
  ratmatrix = csr_matrix((trainevents.rating,(trainevents.user_id,trainevents.item_id)),shape=(len(umap),len(imap)))
  return ratmatrix, umap, imap, testevents

# show the percentage of cells in a (dense) rating matrix that are empty (nan)
def sparsity(arr):
    print("array shape=",arr.shape,"#cells(dense)=","{:,}".format(np.prod(arr.shape)))
    if issparse(arr):
      pcdata = 100 - (len(arr.data)*100.0)/np.prod(arr.shape)
      pcvalid = 100 - (len(arr.data[arr.data != np.nan])*100.0)/np.prod(arr.shape)
      print("%","empty cells= %1.5f (%1.5f incl.na's)" % (pcdata,pcvalid))
      print("numbytes used=", "{:,}".format(arr.data.nbytes + arr.indptr.nbytes + arr.indices.nbytes))
    else:
      #emptypc = (np.isnan(arr).sum()*100.0)/arr.size
      emptypc = float(np.isnan(arr).sum()*100)/np.prod(arr.shape)
      #emptypc = (1.0 - ( count_nonzero(arr) / float(arr.size) )) # alternative, gives same result
      print("%","empty cells (nan's)=%1.5f" % (emptypc)) 
      print("numbytes used=", "{:,}".format(arr.nbytes))

# make user-based CF recommendations for a given target user
def getRecommendations_UU(targetrats, ratsmatrix, imap, simfun=pearsonsim, topN=5, binary=False):
    # get similarity between target and all other users
    sims = []
    for row in ratsmatrix: sims.append(simfun(row,targetrats))
    sims = np.array(sims)
    sims[sims < 0] = np.nan
    # get weighted average of the other users ratings for each unseen movie
    rats = []
    unseenitemidxs = np.where(np.isnan(targetrats)==True)[0]
    if (not binary):
      for col in unseenitemidxs: rats.append(wtavg(ratsmatrix[:,col],sims))
    else:
      for col in unseenitemidxs: rats.append(wtavg(sims,ratsmatrix[:,col]))
    # put results into a dataframe and reverse sort by predicted rating
    itemnames=list(imap.keys())
    rats = pd.DataFrame(rats, index=[itemnames[i] for i in unseenitemidxs], columns=['predrating'])
    rats = rats.sort_values(ascending = False, by=['predrating'])
    return rats[0:min(topN,len(rats))]

# make item-based CF recommendations for a given target user
def getRecommendations_II(targetrats, itemsims, imap, topN=5, binary=False):
    # get wtd average ratings for all the items seen by target (weighted by their similarity to the unseenitems)
    unseenitemidxs = np.where(np.isnan(targetrats)==True)[0]
    seenitems = np.isnan(targetrats)==False
    rats = list([])
    if (not binary):
      for row in unseenitemidxs: rats.append(wtavg(targetrats[seenitems],itemsims[row,seenitems]))
    else:
      for row in unseenitemidxs: rats.append(wtavg(itemsims[row,seenitems],targetrats[seenitems]))
    # put results into a dataframe and reverse sort by predicted rating 
    itemnames=list(imap.keys()) 
    rats = pd.DataFrame(np.array(rats), index=[itemnames[i] for i in unseenitemidxs], columns=['predrating'])
    rats = rats.sort_values(ascending = False, by=['predrating'])
    return rats[0:min(topN,len(rats))]

# compute predicted ratings for the test events (events ~ 'user,item,rating')
def predictRatings(testevents, ratsmatrix, itemsims=False, simfun=cosinesim, binary=False):
    preds = []
    for testevent in testevents:
        print('.', end = '')
        testuser = int(testevent[0])
        testitem = int(testevent[1])
        testuserrats = ratsmatrix[testuser,:]
        testitemrats = ratsmatrix[:,testitem]
        if (type(itemsims) == bool):
          sims = []
          for row in ratsmatrix: sims.append(simfun(row,testuserrats))
          sims = np.array(sims)
          sims[sims < 0] = np.nan
          if (not binary):
            predrat = wtavg(testitemrats,sims)
          else:
            predrat = wtavg(sims,testitemrats)
        else:
          # do item-based CF
          seenitems = np.isnan(testuserrats) == False
          if (not binary):
            predrat = wtavg(testuserrats[seenitems],itemsims[testitem,seenitems])
          else:
            predrat = wtavg(itemsims[testitem,seenitems],testuserrats[seenitems])
        preds.append(predrat)
    return np.array(preds)

def showtypes(df):
  for c in df.columns:
    print(c,type(c[1]),end=" ")

# returns the percentage ranking for each test event
# if itemsims is supplied then do item-based CF, else do user-based CF
# note: the testevents contain user and item names/ids (as defined in the datafile) not matrix indexes
def computePercentageRanking(testevents, ratsmatrix, imap, itemsims=False, simfun=cosinesim, binary=False):
    res = []
    revimap = dict(zip(imap.values(),imap.keys()))
    for testevent in testevents:
        print('.', end = '')
        testuserindx = int(testevent[0])
        if (type(itemsims) == bool):
            recs = getRecommendations_UU(ratsmatrix[testuserindx,:], ratsmatrix, imap, simfun=simfun, topN=1000000, binary=binary)
        else:
            recs = getRecommendations_II(ratsmatrix[testuserindx,:], itemsims, imap, topN=1000000, binary=binary)
        # recs is a dataframe, the row names are the itemnames (as in the datafile)
        # .index() gets the row names, .get_loc returns a row number (starting at 0)
        testitemid = revimap[int(testevent[1])]
        rkpc = ((recs.index.get_loc(testitemid) + 1)*100)/len(recs) 
        res.append(rkpc)
    return np.array(res)

# compute hits and lift for the test events
# if itemsims is supplied then do item-based CF, else do user-based CF
# to work well: number of testevents must be large or else topN has to be large 
def computeLiftOverRandom(testevents, ratsmatrix, imap, itemsims=False, simfun=cosinesim, topN=10, binary=False):
    tothits = randhits = 0 
    revimap = dict(zip(imap.values(),imap.keys()))
    for testevent in testevents:
        print('.', end = '')
        testuserindx = int(testevent[0])
        testitemindx = int(testevent[1])
        if (type(itemsims) == bool):
            recs = getRecommendations_UU(ratsmatrix[testuserindx,:], ratsmatrix, imap, simfun=simfun, topN=topN, binary=binary)
        else:
            recs = getRecommendations_II(ratsmatrix[testuserindx,:], itemsims, imap, topN=topN, binary=binary)
        # getRecommendations() returns actual itemids, hence we need to use revimap below
        if revimap[testitemindx] in recs.index:  
          tothits = tothits + 1
        # do random recommendations
        unseenitemidxs = list(np.where(np.isnan(ratsmatrix[testuserindx,:])==True)[0])
        recs = sample(unseenitemidxs, min(topN,len(recs)))  # only generate same # recs as done above
        if testitemindx in recs:
          randhits =  randhits + 1
    return tothits, randhits 
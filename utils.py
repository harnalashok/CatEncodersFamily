# 24th Dec, 2023

## Utility functions


import pandas as pd
import numpy as np
from scipy.stats import kurtosis
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.feature_selection import  mutual_info_classif
import networkx as nx
# added 27th July, 2023
from cdlib import algorithms
import itertools
import pickle
import pathlib
import string
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import gc




def xgImptFeatures(model, df_columns, filename = None, master=None):
    """
    Given an xgboost classifier model & data col
    names, the function returns two lists of cols,
    one of impt features and the other of 
    unimportant features. Feature importances
    areby 'gain' as against default 'weight'
    in plot_importance() function of xgboost.
   
    Parameters
    ----------
    model: xgboostclassifier model trained on dataframe df
    df_columns: Column names of df. Sequence
                is important: df_columns = list(df.columns)
    master: Folder where impt feature sequence is saved
            File name: fe_1.txt          
    Returns
    -------
    fe_1 : List of columns having features imp > 0 in descending order
    fe_0 : List of columns having features imp of 0.0
    fe   : Pandas Series with indicies as col names against imp scores 
    """
    # Sorted index are in descending order of impt
    sorted_idx = model.feature_importances_.argsort()[::-1]
    fe = pd.DataFrame(
                       model.feature_importances_[sorted_idx],
                       index = df_columns[sorted_idx],
                       columns =['imp']
                     )
    # Columns with zero feature impt
    fe_0 = list(fe.loc[fe['imp'] == 0.0, :].index)
    # Columns greater than than zero feature impt
    # fe_1 is in descending order
    fe_1 = list(fe.loc[fe['imp'] > 0.0, :].index)
    print("Order of feature importance is by 'gain'")
    # Save to file
    if filename is not None:
        # Also save these features
        filename = pathlib.Path(master) / filename
        with open(filename,'w') as tfile:
            tfile.write('\n'.join(fe_1))
    return fe_1, fe_0, fe
   

# Ref: https://www.kaggle.com/code/tetsutani/ps3e16-eda-ensemble-ml-pipeline
# For multiple models
def visualize_importance(models, feature_cols, title, top=9):
    """
    Not tested
    
    Parameters
    ----------
    models : TYPE
        DESCRIPTION.
    feature_cols : TYPE
        DESCRIPTION.
    title : TYPE
        DESCRIPTION.
    top : TYPE, optional
        DESCRIPTION. The default is 9.

    Returns
    -------
    None.

    """
    importances = []
    feature_importance = pd.DataFrame()
    for i, model in enumerate(models):
        _df = pd.DataFrame()
        _df["importance"] = model.feature_importances_
        _df["feature"] = pd.Series(feature_cols)
        _df["fold"] = i
        _df = _df.sort_values('importance', ascending=False)
        _df = _df.head(top)
        feature_importance = pd.concat([feature_importance, _df], axis=0, ignore_index=True)
        
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    # display(feature_importance.groupby(["feature"]).mean().reset_index().drop('fold', axis=1))
    plt.figure(figsize=(12, 4))
    sns.barplot(x='importance', y='feature', data=feature_importance, color='skyblue', errorbar='sd')
    plt.xlabel('Importance', fontsize=14)
    plt.ylabel('Feature', fontsize=14)
    plt.title(f'{title} Feature Importance', fontsize=18)
    plt.grid(True, axis='x')
    plt.show()
    


# https://stackoverflow.com/a/44674459
# ToDO REMOVE zero variance threshold   <=== *****
def remCorrFeatures(data, threshold):
    """
    Desc
    ----
    Function drops features which have pearson's corr
    value more than or equal to the threshold. Absolute
    value of theshold is taken.
    Parameters
    ----------
    data : DataFrame from whom highly corr
           features are to be removed
    threshold : Corr threshold beyond which a 
                feature will be dropped
    Returns
    -------
    un_corr_data : DataFrame with highly corr
                   features dropped
    """
    un_corr_data = data.copy()
    col_corr = set() # Set of all the names of deleted columns
    # Consider both +ve and -ve correlations
    corr_matrix = un_corr_data.corr().abs()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if (corr_matrix.iloc[i, j] >= threshold) and (corr_matrix.columns[j] not in col_corr):
                colname = corr_matrix.columns[i] # getting the name of column
                col_corr.add(colname)
                if colname in un_corr_data.columns:
                    del un_corr_data[colname] # deleting the column from the un_corr_data

    return un_corr_data


def addStatsCols(df):
    """
    Given a dataframe with all numeric features,
    this function creates more numeric features,
    row-wise, as: std, mean, median,min, max
    and kurtosis. Returns modified dataFrame.
    """
    df['std'] = np.std(df,axis = 1)
    df['mean'] = np.mean(df, axis = 1)
    df['median'] = np.median(df,axis = 1)
    df['min'] = np.min(df,axis = 1)
    df['max'] = np.max(df,axis =1)
    df['kurtosis'] = kurtosis(df.values, axis =1 )
    return df



# Generate data using sklearn's make_classification
def transformToCatFeatures(X,y, test_size = 0.25, bins = 20, genColName = True):
    """
    Calls: transformFeatures()
    Called by: main()
    
    Desc
    ----
    The function discretises X assuming all features in X
    have continuous values. Discretisation of all features 
    is perfomed as per number of 'bins' specified. 
    We then have two datasets: original and discretised.
    Both original and discretised versions continue to have 
    the same row-orderings. And hence returned sample of continuous
    data corresponds to returned sample of discrete data.
    
    The functon assumes that X,y have been generated by one of
    sklearn's make datasets algorithm and hence do not have
    header. The function assigns a header.
    
    Parameters
    ----------
    

    Returns
    -------
    orig_train : pandas DataFrame
        Has continuous features
    orig_test : pandas DataFrame
        HAs continuous features
    train_train : pandas DataFrame
        Has discrete features
    train_test : pandas DataFrame
        Has discrete features

    """
    X_data = np.array(X)
    y_data = np.array(y)
    colnames = list(X.columns)
    
    orig_train, orig_test,train_binned,test_binned = transformFeatures(X_data,y_data, test_size, bins, colnames, genColName)
    return orig_train, orig_test, train_binned, test_binned    







# Generate data using sklearn's make_classification
def generateClassificationData(n_samples = 10000, n_features = 10, n_informative = 5,
                               n_classes = 2,n_clusters_per_class = 3, class_sep = 0.5,
                               weights= [0.5,0.5], test_size = 0.25, bins = 20, 
                               flip_y = 0.01, hypercube = True, seed = 42,genColName = True):
    """
    The function generates data for classification task using 
    sklearn's make_classification(). All features generted by 
    make_classification() have continuous values. After data 
    is generated, we discretise all features as per number of
    'bins' specified. We then have two datasets: original and 
    discretised. Both the datasets are shuffled and split into
    train and test data. 
    Note that both the datasets continue to have the same 
    row-orderings. And hence returned sample of continuous
    data corresponds to returned sample of discrete data.
    
    Parameters
    ----------
    For description of parameters, please see sklearn's documentation

    Returns
    -------
    orig_train : pandas DataFrame
        Has continuous features
    orig_test : pandas DataFrame
        HAs continuous features
    train_train : pandas DataFrame
        Has discrete features
    train_test : pandas DataFrame
        Has discrete features

    """
    X, y = make_classification(
                                n_samples=n_samples,  # row number
                                n_features=n_features, # feature numbers
                                n_informative= n_informative, # The number of informative features
                                n_redundant = 0, # The number of redundant features
                                n_repeated = 0, # The number of duplicated features
                                n_classes = n_classes, # The number of classes 
                                n_clusters_per_class= n_clusters_per_class,#The number of clusters per class
                                random_state = 42 ,# random seed
                                class_sep = class_sep,
                                flip_y = flip_y
                             )
    
    X_data = np.array(X)
    y_data = np.array(y)
    
    orig_train, orig_test,train_binned,test_binned = transformFeatures(X_data,y_data, test_size, bins, genColName)
    return orig_train, orig_test, train_binned, test_binned    



 
def transformFeatures(X_data,y_data, test_size = 0.25, bins= 20,  colnames = None, genColName = True):
    
    """
    Called by: transformToCatFeatures() , generateClassificationData
    Calls: None
    
    Desc
    ----
    Given a DataFrame X_data having only continuous features, 
    and y_data the target, the function bins each feature. 
    The resulting dataframe as also the original is partitioned
    into train/test as per specified test_size.
    Parameters
    ----------
    X_data : A pandas DataFrame (data features)
    y_data : A pandas series (data target)
    test_size : How data is to be partitioned into train/test
               The default is 0.25.
    bins : Number of bins into which each column of X
           will be discretised. The default is 20.
    genColName : Should new column names be generated?
                 The default is True.

    Returns
    -------
    orig_train : train sample taken from X 
    orig_test : test sample taken from X
    train_binned : train sample but binned 
                   taken from X
    test_binned : test sample but binned
                  taken from X

    """
    
    if not genColName:
        colnames = colnames
    
    
    orig = pd.DataFrame(X_data)
    train = orig.copy()
    # Generate integer levels
    for i, j in enumerate(train.columns):
        k = ( (i+1) * 4)    # generate an integer as a unique prefix for this iteration
        g = []              # Will contain levels (names)
        # Generate as many level-names as number of bins for jth column
        for i in range(bins):   # For each one of bins (levels)
            g.append(str(k) +str(i+1))   # g conatins a list of names for levels
        
        # for 'j' column, cut it into bins and assign each bin a label 
        # No of bins and o of labels must be same.
        train[j] = pd.cut(train[j], bins, labels = g)  
    
    if genColName:
        # Also generate new column names
        alphabet = string.ascii_lowercase    
        colnames = ["f" + alphabet[i] for i in range(len(train.columns)) ]
        
    train.columns = colnames
    orig.columns = colnames
    train['target'] = y_data
    orig['target'] = y_data
    
    # Shuffle data
    h = train.index.values
    np.random.shuffle(h)
    train = train.loc[h]
    orig = orig.loc[h]
    
    # Pick top 1-test_size rows
    train_binned = train.iloc[ int(train.shape[0] * test_size) :, : ]
    test_binned  = train.iloc[ :int(train.shape[0] * test_size)  , : ]
    
    # Pick bottom test_size rows
    orig_train = orig.iloc[ int(train.shape[0] * test_size) :, : ]
    orig_test = orig.iloc[ :int(train.shape[0] * test_size)  , : ]
    
    return orig_train, orig_test, train_binned, test_binned    
        


# Added on 2nd April, 2023
 # Refer: https://stackoverflow.com/a/31799225
def removeLowVarCols(df,threshold = 0.05, pca = False):
    """
    Remove columns from pandas DataFrame having variance below threshold
    Parameters
    ----------
    df: Pandas DataFrame
    threshold: Float value
    Returns
    -------
    DataFrame with columns having low threshold dropped
    """
    if (pca) :
        pc = PCA(n_components = 0.95)
        out_pc = pc.fit_transform(df)
        c_names = [ "c" + str(i) for i in range(out_pc.shape[1])]
        out_pc = pd.DataFrame(out_pc, columns = c_names)
    else:
         if (threshold != 0):
             out_pc = df.drop(df.std()[df.std() < threshold].index.values, axis=1)
    return out_pc



# Added on 11th April
def binFeatures(df, bins= 20, ):
    for i, j in enumerate(df.columns):
        k = ( (i+1) * 4)    # generate a unique prefix for every iteration
        g = []              # Will contain label names
        for i in range(bins):   # For each one of bins (levels)
            g.append(str(k) +str(i+1))   # g conatins a list of names for levels
            
        df[j] = pd.cut(df[j], bins, labels = g)  
    return df    
    


# rng is randomstate
# rng = np.random.RandomState(123)
def randomSample(ar, perstratasize, rng):
    fact = ar
    idx_arr = np.hstack(
         (
            rng.choice(np.argwhere(fact == 0).flatten(), perstratasize, replace=False),
            rng.choice(np.argwhere(fact == 1).flatten(), perstratasize, replace=False),
            rng.choice(np.argwhere(fact == 2).flatten(), perstratasize, replace=False),
            rng.choice(np.argwhere(fact == 3).flatten(), perstratasize, replace=False),
            rng.choice(np.argwhere(fact == 4).flatten(), perstratasize, replace=False),
            rng.choice(np.argwhere(fact == 5).flatten(), perstratasize, replace=False),
            rng.choice(np.argwhere(fact == 6).flatten(), perstratasize, replace=False),
         )
      )
    tar = np.arange(len(ar))
    tr_ar = list(set(tar).difference(set(idx_arr)))
    return np.array(tr_ar),idx_arr    # train/test arrays






# StackOverflow: https://stackoverflow.com/a/24953575
def plotResults(results, metric):
    """
    Desc
    ----
    
    Plots learning curve after xgboost modeling

    Parameters
    ----------
    results : xgboost results object
    metric : Metric of interest

    Returns
    -------
    None.

    """
    
    ## Plotting results
    epochs = len(results["validation_0"][metric])
    x_axis = range(0, epochs)
    # plot log loss
    fig, ax = plt.subplots(figsize=(12,12))
    ax.plot(x_axis, results["validation_0"][metric], label="Train")
    ax.plot(x_axis, results["validation_1"][metric], label="Test")
    # Set major axis ticks for x and y
    major_x_ticks = np.arange(0, 500, 25)
    major_y_ticks = np.arange(0.9, 1.01, 0.005)
    
    #grid_points = [0.89,0.90,0.91,0.92,0.93,0.94,0.95, 0.955, 0.96,0.97,0.98,0.99,1.0]
    ax.yaxis.set_ticks(major_y_ticks)
    ax.xaxis.set_ticks(major_x_ticks)
    
    ax.grid(True, alpha = 0.5)
    ax.legend()
    plt.ylabel(metric)
    plt.xlabel("No of iterations")
    plt.title("Plot of "+ metric)
    plt.show()



# Plot results of two modeling on the same plot
def plotMulResults(results1, results2, metric, lt_iterations = 500):
    """
    Desc
    -----
    
    Plots learning curves from two different xgboost modeling results

    Parameters
    ----------
    results1 : xgboost results object from Ist modeling
    results2 : xgboost results object from IInd modeling
    metric : Metric being evaluated
    lt_iterations : X-axis iterations limit. The default is 500.

    Returns
    -------
    None.

    """
    ## Plotting results
    epochs = len(results1["validation_0"][metric][:lt_iterations])
    x_axis = range(0, epochs)
    # plot log loss
    fig, ax = plt.subplots(figsize=(12,12))
    ax.plot(x_axis, results1["validation_0"][metric][:lt_iterations], label="Train1")
    ax.plot(x_axis, results1["validation_1"][metric][:lt_iterations], label="Test1")

    ax.plot(x_axis, results2["validation_0"][metric][:lt_iterations], label="Train2")
    ax.plot(x_axis, results2["validation_1"][metric][:lt_iterations], label="Test3")

    # Set major axis ticks for x and y
    major_x_ticks = np.arange(0, lt_iterations, 25)
    major_y_ticks = np.arange(0.9, 1.01, 0.005)
    
    grid_points = [0.89,0.90,0.91,0.92,0.93,0.94,0.95, 0.955, 0.96,0.97,0.98,0.99,1.0]
    ax.yaxis.set_ticks(major_y_ticks)
    ax.xaxis.set_ticks(major_x_ticks)
    
    ax.grid(True, alpha = 0.5)
    ax.legend()
    plt.ylabel(metric)
    plt.xlabel("No of iterations")
    plt.title(f"Plot of {metric} for {lt_iterations} iterations")
    plt.show()
    
    
def bootstrapSample(df)    :
    """
    Desc
    -----
    Return bootstrap samples of
    df, pandas dataframe.
    Parameters
    ----------
    df : Pandas DataFrame

    Returns
    -------
    A bootstrap sample of the dataframe
    
    """
    k = np.arange(df.shape[0])
    rows= np.random.choice(k, size = len(k), replace = True)
    return (df.loc[rows].copy()).reset_index()



# Ref: https://stackoverflow.com/a/4529901/3282777
def savePythonObject(pythonObject, filename, filePath = None ):
    
    """
    Desc
    ----
    Saves any python object to disk
    File is saved to filePath. If
    filePath does not exist, it is
    created.
    Restore the pythonObject
    using restorePythonObject()
    
    Parameters
    ----------
    pythonObject: Obj to be saved
    filename : str Name of pickle file
    filePath: str, NAme of folder
    
    Returns
    -------
    None.
    """
    
    # Current dir is the default filePath
    if filePath is None:
        filePath = pathlib.Path.cwd()
        path = filePath / filename
    else:
        path = pathlib.Path(filePath) / filename
        
    # Check if specified folder exixts:
    # IF not create one:
    p = pathlib.Path(filePath)
    if not pathlib.Path.exists(p):
        print(f"Folder '{filePath}' does not exist")
        print("Being created..")
        p.mkdir(parents=True, exist_ok=True)    
    
    # Now dump the python object    
    with open(path, 'wb') as outp:
        pickle.dump(pythonObject, outp, pickle.HIGHEST_PROTOCOL)
        
    # Finally print your message    
    print("Object saved to:", str(path))    

     
     

 # It restores an earlier saved python object
def restorePythonObject(filename, filePath = None):
     """
     Called by: None
     Calls: None
     Restores an earlier saved python object
     in pickle format by savePythonObject()
     Parameters
     ----------
     filename : Pickle file having python-object
     filePath : Folder where filename is placed
                Default is modelsPath

     Returns
     -------
     Python-object

     """
     if filePath is None:
        filePath = pathlib.Path.cwd()
        path = filePath / filename
        print(f"Looking for '{filename}' in folder: {str(filePath)}" )
     else:
        path = pathlib.Path(filePath) / filename

     with open(path, 'rb') as inp:
         ct = pickle.load(inp)
     return ct    



def pcaAndConcat(vec_train, vec_test, n_components = 2, scaleData = True):
    """
    Calls: 
    Called by: main method as needed    
    
    
    Desc
    ----
        Given a dictionray with keys as respective cat-columns 
        and the values as dataframes (of unitvectors), the function 
        performs PCA on each value (ie. the dataframe) and outputs 
        one concatenated DataFrame. This it does for both train/test
        dictionaries. 
    
    Example: 
        Assume our original training data had 5-cat columns. These
        5-cat-cols would be keys of dict: vec_train and vec_test. 
        Corresponding to every key the input dict will have 
        5-dataframes of unit vectors as values. No of rows in each 
        dataFrame would be the same, that is, as many as in train/test 
        data. We then take PCA of each dataframe and reduce the 
        unit vectors to n_components (default is 2 columns). Thus, 
        we will have 5 X n_components in all. We concatenate these 
        5 X n_components (or columns) to make our training/test data.
    
    Parameters
    ----------
    
    vec_train: Dictionary of Dataframes with keys as cat-cols.
               It would contain unit vectors per-level per-row 
               for each cat col of train data.
               
    vec_test:  Same as vect_train but for tes data        
               
    n_components: int, No of PCA components to reduce each DataFrame to.
                  Default is 2.
    scaleData: boolean; Should data be centered and scaled before PCA?
               Default is True.
    
    Returns
    -------
    
    Concated dataframes and two dictionaries. Each dictionary is of 
    DataFrames, as before, except that evry DataFrame that was input
    is replaced by its PCA version with as many columns as n_components.
    Two dictionaries returned are mainly for debugging purposes.

    """
    
    # Do not alter original dataframes
    vt = vec_train.copy()
    ve = vec_test.copy()

    # What all columns exist?
    rt = list(vt.keys())
    
    # For every col in rt
    # StandardScale before PCA
    if scaleData:
        for key in rt:
            print(f"Performing PCA for column, {key} , in train data")
            # Instantiate PCA
            pca = PCA(n_components = n_components)
            ss = StandardScaler()
            # Perform PCA of train DataFrame of unit vectors
            k = pca.fit_transform(ss.fit_transform(vec_train[key]))
            vt[key] = pd.DataFrame(k, columns = ['pc' + key+ str(i) for i in range(n_components)])
            print(f"Performing PCA for column, {key} , in test data")
            # Transform test data now
            k = pca.transform(ss.transform(vec_test[key]))
            ve[key] = pd.DataFrame(k, columns = ['pc' + key +str(i) for i in range(n_components)])
    else:
        for key in rt:
            print(f"Performing PCA for column, {key} , in train data")
            # Instantiate PCA
            pca = PCA(n_components = n_components)
            # Perform PCA of train DataFrame of unit vectors
            k = pca.fit_transform(vec_train[key])
            vt[key] = pd.DataFrame(k, columns = ['pc' + key+ str(i) for i in range(n_components)])
            print(f"Performing PCA for column, {key} , in test data")
            # Transform test data now
            k = pca.transform(vec_test[key])
            ve[key] = pd.DataFrame(k, columns = ['pc' + key +str(i) for i in range(n_components)])
    
    obj_tr = [ vt[rt[i]]  for i in range(len(rt))]
    obj_te = [ ve[rt[i]]  for i in range(len(rt))]
    # Amended 24th Dec, 2023
    print("Concatenating PCAs of train data")
    cc_tr = pd.concat(obj_tr, axis = 1)
    print("Concatenating PCAs of test data")
    cc_te = pd.concat(obj_te, axis = 1)
    print("Done......")
    return cc_tr,cc_te,  vt ,ve, 



# Ref: https://www.kaggle.com/code/ryanholbrook/mutual-information
def calMIScores(X, y):
    """
    Desc
    ---- 
    Calculates mutual information scores when target
    is discrete.

    Parameters
    ----------
    X : Pandas DataFrame 
    y : An array or Series; target

    Returns
    -------
    mi_scores : Pandas Series (sorted)

    """
    
    # Get sores
    mi_scores = mutual_info_classif(X, y)
    # Transform mi_scores to a Series:
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    # Sort Series in Descending order:
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores



def plotSeries(scores, title= ""):
    """
    Desc
    ----
    Plots MI scores or feature impt by centrality 
    calculated by calMIScores() or by featureImptByCentrality()
    Use it as:
        plt.figure(dpi=100, figsize=(8, 5))
        plotSeries(calMIScores(X, y), someTitle)
        OR, as
        plotSeries(featureImptByCentrality, someTitle)
        
    Parameters
    ----------
    scores: Pandas Series
    title: Graph title; default: Empty str
    
    """
    # Scores sorted in ascending order:
    scores = scores.sort_values(ascending=True)
    # Map parameters:
    width = np.arange(len(scores))
    # Write ticks as per scores index:
    ticks = list(scores.index)
    # Plot width:
    plt.barh(width, scores)
    # Plot yticks
    plt.yticks(width, ticks)
    plt.title(title)
    
    

def featureImptByCentFeatCounts(colList, normalize = False):
    """
    Desc
    ----
    In the colList, how many columns pertain to
    degree centrality, eigenvector centrality,
    page-rank and clustering coeff. Plot
    by using plotSeries().
    
    Example:
        Here is a table of col-name wise xgboost impt score
        
        deg_abc   0.01
        pr_cde    0.02
        pr_def    0.015
        eig_xy    0.11
        deg_ff    0.001
        deg_uy    0.01
        
    Result (normalized):
        
        degree      3/6
        pagerank    2/6
        eigenvector 1/6

    Parameters
    ----------
    colList : A list of columns
    normalize: Boolean

    Returns
    -------
    d : A sorted pandas Series

    """
    d = {'degree' : 0, 'pagerank' : 0, 'eigenvector' : 0, 'clusteringcoeff' : 0,
         "betweenness" : 0, 'avgembeddedness' : 0, 'leidencomsdensity' : 0}
    deg = [i for i in colList if 'deg_' in i ]
    d['degree'] = len(deg)
    pr =  [i for i in colList if 'pr_' in i ]
    d['pagerank'] = len(pr)       
    eig = [i for i in colList if 'eig_' in i ] 
    d['eigenvector'] = len(eig)      
    clu = [i for i in colList if 'clu_' in i ]
    d['clusteringcoeff'] = len(clu)
    bet = [i for i in colList if 'bet_' in i ]
    d['betweenness'] = len(bet)
    ae = [i for i in colList if 'ae_' in i ]
    d['avgembeddedness'] = len(ae)
    den = [i for i in colList if 'den_' in i ]
    d['leidencomsdensity'] = len(den)
    # Transform d to pandas Series:
    d = pd.Series(d).sort_values(ascending = False)
    if (normalize):
        d = d/sum(d)
    return d



def featureImptByScore(score, colList, normalize = False):
    """
    Desc
    ----
    In the colList, total score of columns
    that pertain to degree centrality, to eigenvector
    centrality, to page-rank and clustering coeff and 
    others. Plot by using plotSeries().
    
    Example:
        Here is a table of col-name wise xgboost impt score
        
        deg_abc   0.01
        pr_cde    0.02
        pr_def    0.015
        eig_xy    0.11
        deg_ff    0.001
        
    Result:
        degree      0.011
        pagerank    0.035
        eigenvector 0.11

    Parameters
    ----------
    score: Pandas Series giving column wise impt score
    colList : A list of dataframe columns
    normalize: Boolean

    Returns
    -------
    d : A sorted pandas Series

    """
    d = {'degree' : 0, 'pagerank' : 0, 'eigenvector' : 0, 'clusteringcoeff' : 0,
         "betweenness" : 0, 'avgembeddedness' : 0, 'leidencomsdensity' : 0}
    deg = [i for i in colList if 'deg_' in i ]
    d['degree'] = score[deg].sum()
    pr =  [i for i in colList if 'pr_' in i ]
    d['pagerank'] = score[pr].sum()      
    eig = [i for i in colList if 'eig_' in i ] 
    d['eigenvector'] = score[eig].sum()    
    clu = [i for i in colList if 'clu_' in i ]
    d['clusteringcoeff'] = score[clu].sum()
    bet = [i for i in colList if 'bet_' in i ]
    d['betweenness'] = score[bet].sum()
    ae = [i for i in colList if 'ae_' in i ]
    d['avgembeddedness'] = score[ae].sum()
    den = [i for i in colList if 'den_' in i ]
    d['leidencomsdensity'] = score[den].sum()
    # Transform d to pandas Series:
    d = pd.Series(d).sort_values(ascending = False)
    if (normalize):
        d = d/sum(d)
    return d
 

      
       
def plotBipartiteGraph(filename, pathToFolder, ax=None , title = None, node_size = 30, connected = False, takeGraphSample = False, noSampleNodes=100):
    """
    Plots a bipartitie graph.  
    
    Parameters
    ----------
    filename : Filename
    pathToFolder : Folder where file resides
    ax: Matplotlib axis
    connected: Boolean; Show only connected nodes
               Default is False
    takeGraphSample: boolean; Take a sample of graph for plotting
    noSampleNodes: int; No of nodes to be taken if takeGraphSample is True 
          
    Returns
    -------
    None.

    """
    filepath = pathToFolder / filename
    G = nx.read_gml(filepath) 
    
    # Ref: https://stackoverflow.com/a/62407941/3282777
    if takeGraphSample:
        sampled_nodes = np.random.choice(G.nodes, noSampleNodes)
        G = G.subgraph(sampled_nodes)
    
    if connected:
      gcc = max(nx.connected_components(G), key=lambda x: len(x))
      # subgraph of connected nodes:
      H = G.subgraph(gcc)
      GH = H
    else:
      GH = G  
      
    color_dict = {0:'b',1:'r'}
    color_list = [color_dict[i[1]] for i in GH.nodes.data('bipartite')]
    color_list
    pos = nx.spring_layout(GH, k = None, scale = 4, iterations = 1000)
    #plt.figure(figsize= (10,5));
    #ax = plt.gca();
    nx.draw(GH, 
            pos= pos,
            node_size= node_size,
            with_labels =  False,
            font_size = 10,
            node_color = color_list,
            ax =ax,
            width = 0.1  # edge width
            );
    
    fontx = {'color':'green','size':8}
    plt.title(title, fontdict = fontx);




def plotUnipartiteGraph(filename, pathToFolder, ax = None,title = None, node_size = 30, connected=False, takeGraphSample = False, noSampleNodes=100):
    """
    Plots a unipartite graph
    Parameters
    ----------
    filename : graph file.
    pathToFolder : Path to folder of graph file.
    takeGraphSample: boolean; Take a sample of graph for plotting
    noSampleNodes: int; No of nodes to be taken if takeGraphSample is True 

    Returns
    -------
    None.

    """
    filepath = pathToFolder / filename
    G = nx.read_gml(filepath) 
    
    # If network is very large, take a sanple
    # Ref: https://stackoverflow.com/a/62407941/3282777
    if takeGraphSample:
        sampled_nodes = np.random.choice(G.nodes, noSampleNodes)
        G = G.subgraph(sampled_nodes)
    
    if connected:
        gcc = max(nx.connected_components(G), key=lambda x: len(x))
        # subgraph of connected nodes:
        H = G.subgraph(gcc)
        GH = H
    else:
        GH = G
    pos = nx.spring_layout(GH, k = None)
    #plt.figure(figsize= (10,5));
    #ax = plt.gca();
    nx.draw(GH, 
            pos= pos,
            node_size = node_size,
            with_labels =  False,
            font_size = 10,
            ax =ax,
            width = 0.1  # edge width
            );

    fontx = {'color':'green','size':8}
    plt.title(title, fontdict = fontx);
   

    
# Community visualization
# Kaggle: https://www.kaggle.com/code/rahulgoel1106/commmunity-detection-using-igraph-and-networkx
def communityVisualization(filename, pathToFolder, algo = nx.community.greedy_modularity_communities ,
                           ax= None, title = None, withLabels = False, font_size = 8, edgewidth = 0.1,
                           node_size = None, k = None, colorList =  ["orange","yellow","cyan","green","red", "purple"]*20,
                           takeGraphSample = False, noSampleNodes=100):
    """
    Desc
    ----
    Displays communities created by given community algo.

    Parameters
    ----------
    filename : str, Graph file
    pathToFolder : str, Path to folder having graph files
    algo: function object to create communities
          Default: nx.community.greedy_modularity_communities 
          Examples: nx.community.kernighan_lin_bisection
                    nx.community.greedy_modularity_communities
                    nx.community.louvain_communities
                    algorithms.leiden  (from cdlib library)
    ax: Matplotlib axis object
    withLabels : boolean, Should labels be displayed? The default is False.
    font_size : int, Label font size. The default is 8.
    edgewidth: float, Edge line width
    node_size: int
    k: float; optimal distance between nodes. 
    colorList: List of colors for difft communitied. 
               Default is: ["orange","yellow","cyan","green","red","purple"]*20

    Returns
    -------
    None.

    """
    filepath = pathToFolder / filename
    G = nx.read_gml(filepath) 
    if (len(G.edges())) == 0: 
        print("This network has no edges. Moving ahead.")
        return 

    
    # If network is very large, take a sanple
    # Ref: https://stackoverflow.com/a/62407941/3282777
    if takeGraphSample:
        sampled_nodes = np.random.choice(G.nodes, noSampleNodes)
        G = G.subgraph(sampled_nodes)
        if (len(G.edges())) == 0: 
            print("This network has no edges. Moving ahead.")
            return 
    
    colors = colorList
    pos = nx.spring_layout(G, k)
    
    if algo == nx.community.girvan_newman:
        lst_b = algo(G)
        lst_b = tuple(sorted(c) for c in next(lst_b))
        lst_b = [frozenset(i) for i in lst_b ]
    # Changed on 27th July, 2023    
    if algo ==  algorithms.leiden:
        lst_b = algo(G)
        lst_b = [frozenset(i) for i in lst_b.communities ]
    else:
        lst_b = algo(G)    

    
    color_map_b = {}
    keys = G.nodes()
    values = "black"
    for i in keys:
            color_map_b[i] = values
    counter = 0
    for x in lst_b:
      for n in x:
        color_map_b[n] = colorList[counter]
      counter = counter + 1
    nx.draw_networkx_edges(G, pos, width = edgewidth, ax =ax);
    
    if node_size is not None:
        nx.draw_networkx_nodes(G, pos, node_size = node_size, node_color=dict(color_map_b).values(), ax =ax);
    else:
        nx.draw_networkx_nodes(G, pos,  node_color=dict(color_map_b).values(), ax =ax);
    if withLabels:
      nx.draw_networkx_labels(G, pos, font_size = font_size, font_weight = 'bold',ax =ax)
    plt.axis("off");
    
    fontx = {'color':'green','size':8}
    plt.title(title, fontdict = fontx);

    #plt.show();    




def transformBinnedDF2Communities(columnNames,pathToGraphFolder, train_binned, algo=nx.community.greedy_modularity_communities):
    """
    Desc
    -----
    Replace every column in the binned the Dataframe as per network community
    Example binned/discrete dataframe:
            'a'     'b
            --      ---
            434     709
            435     789
            23      710
            78      756

    Replace col 'a' as per communities discovered in network 'a_projected_b.gml':
    434 & 435 belong to one community and 23 & 78 to other.    

            'a'     'b'
            ---     ---
             0      709
             0      789
             1      710
             1      756

    Similarly, replace col 'b' as per communities in 'b_projected_a.gml':

            'a'     'b'
            ---     ---
             0      1
             0      0
             1      0
             1      1
             
    Parameters
    -----------     
    columnNames: list of cat cols
    pathToGraphFolder: str: Folder that has graph files     
    train_binned: DataFrame with binned/discrete columns  
    algo: function object: networkx algorithm to be used to 
          discover communities.
          Examples: nx.community.kernighan_lin_bisection
                    nx.community.greedy_modularity_communities
                    nx.community.louvain_communities

          
    Returns
    -------
    map_list: list of dictionaries. Each dictionary maps
              levels in a column to a community 
    df: DataFrame: Transformed train_binned DataFrame.
        If train_binned had columns other than columnNames,
        df would not contain those columns          


    """

    # 1. From col-names names, create graph filenames:
    # Example: col-names: ['a','b','c']
    # First permute col-names in pairs of two:
    #  p = ('a','b'),('b','a'), ('a','c'),('c','a')...
    p = list(itertools.permutations(columnNames,2))
    # Prepare list of graph files:
    # ('a_projected_b.gml', 'b_projected_a.gml'...)
    filelist = [] 
    for i in p:
      filelist.append(i[0] + "_projected_" + i[1] + ".gml")
    
    # Check if all files in filelist exist. Or which files exist?      
    filesexist = []
    for i in filelist:
        myfile = Path(pathToGraphFolder) / i
        if myfile.exists():
            filesexist.append(i) 
            
    filelist = filesexist        

    # 2. For every file in the filelist
    map_list = []  # Start with a blank list of dictionaries
    # df will store dataframe train_binned after each bin-col is mapped
    #  
    df = pd.DataFrame()
    for file in filelist:
      print("Reading file: ", file)
      # 2.1 Load network
      # Get full filename that includes filepath
      filepath = Path(pathToGraphFolder) / file
      # Read the file as a network
      G = nx.read_gml(filepath)
      if (len(G.edges())) == 0: 
        print("This network has no edges. Moving ahead.")
        continue 
       
      # 2.2 Calculate community classes using algo
      #    cm_mod contains as many frozensets of nodes
      #    as there are communities:
          
      if algo == nx.community.girvan_newman:
          cm_mod = algo(G)
          cm_mod = tuple(sorted(c) for c in next(cm_mod))
          cm_mod = [frozenset(i) for i in cm_mod ]
      else:
          cm_mod = algo(G)
          #print(cm_mod)

      # 3.0 We now create dict corresponding to
      #     all communities in cm_mod
      #    Example:
      #                 frozenset1         frozenset2
      #   cm_mod:     [{'434', '435' },    {'23' , '78'}]
      #   fset_dict  {'434': 0, '435' :0, '23' : 1, '78': 1}

      counter = 0  # Assigns values in dict
      fset_dict = {}  # Start with blank dict
      # For every frozenset
      for i in cm_mod:
        # If set i is not a frozenset, make it so
        # else its members cannot be used as dict keys
        if not isinstance(i,frozenset):
            i = frozenset(i)
            
        # For every item in this frozenset
        for g in i:
          # Set class to the value of counter
          fset_dict[g] = counter

        # Increment counter for next class
        counter  +=1
      # Now that map dict for the modularity
      # classes are ready, append it to map_list
      map_list.append(fset_dict)

      # Extract column name from file:
      colToProject = file.split('_')[0]
      # Map train_binned column using the dict
      df[file] = (train_binned[colToProject].astype(str)).map(fset_dict)
      # Continue for loop for the next filelist
    return map_list,df




def transformBinnedDF2CommunitiesAll(pathToGraphFolder, train_binned, algo):
    """
    Desc
    -----
    Search out all graph files and Replace every column in the binned Dataframe 
    as per network community
    Example binned/discrete dataframe:
            'a'     'b
            --      ---
            434     709
            435     789
            23      710
            78      756

    Replace col 'a' as per communities discovered in network 'a_projected_b.gml':
    434 & 435 belong to one community and 23 & 78 to other.

            'a'     'b'
            ---     ---
             0      709
             0      789
             1      710
             1      756

    Similarly, replace col 'b' as per communities in 'b_projected_a.gml':

            'a'     'b'
            ---     ---
             0      1
             0      0
             1      0
             1      1

    Parameters
    -----------
    pathToGraphFolder: str: Folder that has graph files
    train_binned: DataFrame with binned/discrete columns
    algo: function object: networkx algorithm to be used to
          discover communities.
          Example: nx.community.greedy_modularity_communities

    Returns
    -------
    map_list: list of dictionaries. Each dictionary maps
              levels in a column to a community
    df: DataFrame: Transformed train_binned DataFrame.
        If train_binned had columns other than columnNames,
        df would not contain those columns


    """
    # 15.1 Path where .gml files are placed:

    # 1. From col-names names, create graph filenames:
    filelist = sorted(list(Path(pathToGraphFolder).iterdir()))
    filelist = [i   for i in filelist if "_projected_" in str(i) ]
    filelist = [str(i).split("/")[-1]   for i in filelist ]

    # 2. For every file in the filelist
    map_list = []  # Start with a blank list of dictionaries
    # df will store dataframe train_binned after each bin-col is mapped
    #
    df = pd.DataFrame()
    for file in filelist:
      # 2.1 Load network
      # Get full filename that includes filepath
      filepath = Path(pathToGraphFolder) / file
      # Read the file as a network
      G = nx.read_gml(filepath)

      # 2.2 Calculate community classes using algo
      #    cm_mod contains as many frozensets of nodes
      #    as there are communities:
          
      if algo == nx.community.girvan_newman:
          cm_mod = algo(G)
          cm_mod = tuple(sorted(c) for c in next(cm_mod))
          cm_mod = [frozenset(i) for i in cm_mod ]
      else:
          cm_mod = algo(G)

      # 3.0 We now create dict corresponding to
      #     all communities in cm_mod
      #    Example:
      #                 frozenset1         frozenset2
      #   cm_mod:     [ {'434', '435' },    {'23' , '78'}]
      #   fset_dict  {'434': 0, '435' :0, '23' : 1, '78': 1}

      counter = 0  # Assigns values in dict
      fset_dict = {}  # Start with blank dict
      # For every frozenset
      for i in cm_mod:
        if not isinstance(i,frozenset):
          i = frozenset(i)
        # For every item in this frozenset
        for g in i:
          # Set class to the value of counter
          fset_dict[g] = counter

        # Increment counter for next class
        counter  +=1
      # Now that map dict for the modularity
      # classes are ready, append it to map_list
      map_list.append(fset_dict)

      # Extract column name from file:
      colToProject = file.split('_projected_')[0]
      # Map train_binned column using the dict
      if colToProject in train_binned.columns:
        df[file] = train_binned[colToProject].map(fset_dict)
      # Continue for loop for the next filelist
    return map_list,df
    


# Function to remove const columns from DataFrame
# https://stackoverflow.com/a/20210048
def remConstColumns(df):
    """
    Desc
    ----
    Removes const columns from dataframe df

    Parameters
    ----------
    df : pandas DataFrame

    Returns
    -------
    df : pandas DataFrame

    """
    print(f"Datashape before processing: {df.shape}")
    df = df.loc[:, (df != df.iloc[0]).any()] 
    print(f"Datashape after processing: {df.shape}")
    return df



def featureEngKmeans(train, test, n_clusters,step_size, km = True, gmm = True, random_state = 42):
    """
    Parameters
    ----------
    train : DataFrame
    test : DataFrame
    n_clusters : int; Number of clusters
    step_size : int; No of cat_cols
    km : boolean; Should features be generated through KMeans
         Defaul.
    gmm : boolean; Should features be generated through GMM
    random_state : int;  The default is 42.

    Returns
    -------
    tr : DataFrame
    te : DataFrame
    """
    
    if (not km and not gmm):
        return train,test
    
    tr = train.copy()
    te = test.copy()
    label_tr = []
    label_te = []
    for i in range(0,tr.shape[1],step_size):
      print("Current index: ", i)
      if km:
        km = KMeans(n_clusters = n_clusters,random_state = random_state)
        _= km.fit_transform(tr.iloc[:, i:i+step_size].values.astype(np.float))
        label_tr.append(km.labels_)
        label_te.append(km.predict(te.iloc[:,i:i+step_size]))

      if gmm:
        gmm = GaussianMixture(n_components = n_clusters, random_state = random_state)
        gmm.fit(tr) 
        label_tr.append(gmm.predict(tr))
        label_te.append(gmm.predict(te))

      _=gc.collect()

    for i in range(len(label_tr)):
      tr['clu'+str(i)] = label_tr[i]
      te['clu' + str(i)] = label_te[i] 

    return tr,te



################### That's all folks #######################




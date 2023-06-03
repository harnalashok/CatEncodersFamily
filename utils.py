# 3rd June, 2023

## Utility functions


import pandas as pd
import numpy as np
from scipy.stats import kurtosis
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
import pickle
import pathlib
import string
import matplotlib.pyplot as plt



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
    return fe_1, fe_0
   



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
def generateSklearnData(X,y, test_size = 0.25, bins = 20, genColName = True):
    """
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
    
    orig_train, orig_test,train_binned,test_binned = transformToCatFeatures(X_data,y_data, test_size, bins, colnames, genColName)
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
    
    orig_train, orig_test,train_binned,test_binned = transformToCatFeatures(X_data,y_data, test_size, bins, genColName)
    return orig_train, orig_test, train_binned, test_binned    


 
def transformToCatFeatures(X_data,y_data, test_size = 0.25, bins= 20,  colnames = None, genColName = True):
    
    """
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
    Saves any python object to disk
    File is saved to filePath. Restore it
    using restorePythonObject()
    Parameters
    ----------
    filename : pickle file for dumping
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
    with open(path, 'wb') as outp:
        pickle.dump(pythonObject, outp, pickle.HIGHEST_PROTOCOL)

     
     

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
    Given a dictionray of dataframes, the function performs PCA
    on each dataframe and outputs a concatenated dataframe. 
    This it does for both the dictionaries and outputs two dataframes. 
    
    Parameters
    ----------
    
    vec_train: Dictionary of Dataframes. It would contain unit vectors
               for each cat col of train data.
               
    vec_test:  Dictionary of Dataframes. It would contain unit vectors
               for each cat col of test data.         
               
    n_components: No of PCA components. Default is 2.
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
            print(f"Performing PCA for {key} for train data")
            # Instantiate PCA
            pca = PCA(n_components = n_components)
            ss = StandardScaler()
            # Perform PCA of train DataFrame of unit vectors
            k = pca.fit_transform(ss.fit_transform(vec_train[key]))
            vt[key] = pd.DataFrame(k, columns = ['pc' + key+ str(i) for i in range(n_components)])
            print(f"Performing PCA for {key} for test data")
            # Transform test data now
            k = pca.transform(ss.transform(vec_test[key]))
            ve[key] = pd.DataFrame(k, columns = ['pc' + key +str(i) for i in range(n_components)])
    else:
        for key in rt:
            print(f"Performing PCA for {key} for train data")
            # Instantiate PCA
            pca = PCA(n_components = n_components)
            # Perform PCA of train DataFrame of unit vectors
            k = pca.fit_transform(vec_train[key])
            vt[key] = pd.DataFrame(k, columns = ['pc' + key+ str(i) for i in range(n_components)])
            print(f"Performing PCA for {key} for test data")
            # Transform test data now
            k = pca.transform(vec_test[key])
            ve[key] = pd.DataFrame(k, columns = ['pc' + key +str(i) for i in range(n_components)])
    
    obj_tr = [ vt[rt[i]]  for i in range(len(rt))]
    obj_te = [ ve[rt[i]]  for i in range(len(rt))]
    print("Concatenating train data")
    cc_tr = pd.concat(obj_tr, axis = 1)
    print("Concatenating test data")
    cc_te = pd.concat(obj_te, axis = 1)
    print("Done......")
    return cc_tr,cc_te,  vt ,ve, 



############################ BEGIN ###################




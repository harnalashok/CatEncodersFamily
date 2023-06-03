"""
30th May, 2023
PCA gives good results after few tries

Data set: https://www.kaggle.com/datasets/vuppalaadithyasairam/kidney-stone-prediction-based-on-urine-analysis

Try:
    
Mobile price classification
https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification?select=train.csv    


Try 
VarianceThreshold(threshold=0.01)
sel = VarianceThreshold(threshold=0.01)  # 0.1 indicates 99% of observations approximately
sel.fit(X_train)  # fit finds the features with low variance

# get_support is a boolean vector that indicates which features 
# are retained. If we sum over get_support, we get the number
# of features that are not quasi-constant
len(X_train.columns[sel.get_support()])


Objective: The basic objective is to show that
           we can rely on our engineered features 
           to look at the clusters.


"""

%reset -f

# 1.0 Call libraries
import pandas as pd
import numpy as np



# 1.01
from sklearn.datasets import make_blobs,make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
#import umap  # Takes long time to import
import seaborn as sns
import xgboost as xgb
from xgboost import plot_importance

from catencfamily.catencfamily import CatEncodersFamily


# 1.02 Misc
import os,gc , time


# 1.03 Home made modules

os.chdir("C:\\Users\\Ashok\\OneDrive\\Documents\\talkingdata\\20042023_breastcancer\\")
import utils


# 1.04
#import importlib; importlib.reload(network_features)
import importlib; importlib.reload(utils)
#import importlib; importlib.reload(catfamilyenc)




# 1.05
dataPath =                 "C:\\Users\\Ashok\\OneDrive\\Documents\\talkingdata\\20042023_breastcancer\\"
modelsPath =               "C:\\Users\\Ashok\\OneDrive\\Documents\\talkingdata\\20042023_breastcancer\\allmodels\\models\\"
pathToStoreProgress =      "C:\\Users\\Ashok\\OneDrive\\Documents\\talkingdata\\20042023_breastcancer\\allmodels\\progress\\"
master =  dataPath + "master\\"





os.chdir(dataPath)

# 2.0 Decide program-wide seed
rng = np.random.default_rng(1451)
seed = rng.integers(low = 30,high = 90000 )
seed




# Read Data
data = pd.read_csv("kidney_stone.csv")

data.head()
data.shape     # (569, 33)
data.columns
y = data.pop("target")
data.pop("id")

# Check nulls
data.isnull().sum()



data.columns


# Discretise all features of data. And also split dataset.
# into train/test
orig_train, orig_test, train_binned, test_binned  = utils.generateSklearnData(data,
                                                                              y,
                                                                              bins = 30,
                                                                              test_size = 0.2,
                                                                              genColName = False  # Keep orig col names
                                                                              )

                                                                                  




# Check  original data
orig_train.head()
orig_test.head()
orig_train.shape   # (513,31)

# Check discretised features
train_binned.columns
train_binned.head()
test_binned.head()


## Save aour dataframes as we may need them later
os.chdir(master)

# orig data contains target as the last column
# WHEN index is not saved (index = False) 
# then on reading back index is reset and 
# begins from 0
orig_train.to_csv("orig_train.csv", index = False)
orig_test.to_csv("orig_test.csv", index = False)

# Binned data contains target as the last column
train_binned.to_csv("train_binned.csv", index = False)
test_binned.to_csv("test_binned.csv", index = False)

##########################
## Read back org saved data 
###########################
os.chdir(master)
orig_train = pd.read_csv("orig_train.csv")
orig_test= pd.read_csv("orig_test.csv")
train_binned = pd.read_csv("train_binned.csv")
test_binned= pd.read_csv("test_binned.csv")


# Just recheck/explore
train_binned.columns
train_binned.dtypes   # All int64
train_binned.head()
test_binned.head()



# Pop out targets
ytr = train_binned.pop('target')
yte = test_binned.pop("target")


## Developing models
# Which are our cat columns
# We will consider few columns
# Ref: https://www.kaggle.com/code/kanncaa1/feature-selection-and-data-visualization
#best_features= ["area_mean", "area_se", "texture_mean", "concavity_worst", "concavity_mean"]
best_features = ['Glucose','BMI','Age','Insulin','DiabetesPedigreeFunction']
cat_cols = list(train_binned.columns)
cat_cols  = best_features
len(cat_cols)  # 6




# Keep interacting columns, same
# as cat columns:
interactingCatCols = cat_cols

# Instantiate CustomTransformer class:




ct = CatEncodersFamily(pathToStoreProgress, 
                       modelsPath,
                       cMeasures=[1,1,1,1,None,1,1],    # Better clusters
                       #cMeasures=[0,1,0,0,None,0,0],    # GOOD clusters
                       noOfColsToConcat = 2
                       )

# Fit it:
ct.fit(train_binned, cat_cols, interactingCatCols) 
utils.savePythonObject(ct, "abc.pkl")

# Transform test_binned data with only cat_cols
out = ct.transform(test_binned[cat_cols])
out.shape      #  (56,373)
out.columns
# Remove low variance columns
#out = utils.removeLowVarCols( out , pca = False)
out.shape   #  (56, 207)
yte.shape   # 
out.columns

os.chdir(master)
out.to_pickle("test_transformed.pkl")
yte.to_pickle("yte_unprocessed.pkl")



# Check list of original columns
gc.collect()
out_tr = ct.transform(train_binned[cat_cols])
out_tr.shape

os.chdir(master)
out_tr.to_pickle("train_transformed.pkl")
ytr.to_pickle("ytr_unprocessed.pkl")




##############################
## Start reading
#############################

os.chdir(master)
train_trans = pd.read_pickle("train_transformed.pkl")
test_trans = pd.read_pickle("test_transformed.pkl")
y_train = pd.read_pickle("ytr_unprocessed.pkl")
y_test = pd.read_pickle("yte_unprocessed.pkl")

train_trans.shape   #  (513, 207)
test_trans.shape    #  (56, 207)


# Read binned data 
os.chdir(master)
train_binned = pd.read_csv("train_binned.csv")
test_binned= pd.read_csv("test_binned.csv")

train_binned.shape  # (513, 31)
test_binned.shape  # (56, 31)



# Read org data 
orig_train = pd.read_csv("orig_train.csv")
orig_test= pd.read_csv("orig_test.csv")

orig_train.shape  # (513,31)
orig_test.shape   # (56, 31)

# poput targets
u = orig_train.pop('target')
ue = orig_test.pop('target')
train_binned.pop('target')
test_binned.pop('target')



# Reset index. Step is redundant
orig_train = orig_train.reset_index(drop = True)
orig_test = orig_test.reset_index(drop =True)
train_binned = train_binned.reset_index(drop = True)
test_binned = test_binned.reset_index(drop =True)

orig_train.head()
orig_test.head()
train_binned.head()
test_binned.head()

# Merge now : orig + binned. Columns will double
# We will take only the best features while merging:
org_binned_train = pd.merge(orig_train[cat_cols], train_binned[cat_cols], left_index=True, right_index=True )
org_binned_test = pd.merge(orig_test[cat_cols], test_binned[cat_cols], left_index=True, right_index=True )


# Check shapes
org_binned_train.shape  # (513,10)
org_binned_test.shape   #  (56,10)  


#### 
# Read transformed data
train_trans = pd.read_pickle("train_transformed.pkl")
test_trans = pd.read_pickle("test_transformed.pkl")
y_train = pd.read_pickle("ytr_unprocessed.pkl")   # For both orig and binned train
y_test = pd.read_pickle("yte_unprocessed.pkl")    # For both orig and binned test


# Reset Index.  Step is redundant
train_trans = train_trans.reset_index(drop = True)
test_trans = test_trans.reset_index(drop = True)

train_trans.columns[6:]


# Merge now:
# So we have 20 initial columns of same name    
org_trans_train = pd.merge(orig_train[cat_cols], train_trans, left_index=True, right_index=True )
org_trans_test = pd.merge(orig_test[cat_cols], test_trans, left_index=True, right_index=True )

# Check shapes
org_trans_train.shape   #  (513, 212)
org_trans_test.shape    #  (56, 212)
org_trans_train.columns[20:]


# Reset index for y_train/y_test
# Redundant step. But safe.
gc.collect()
y_train = y_train.reset_index(drop = True)
y_test = y_test.reset_index(drop = True)

# Check shapes and values
y_train
y_test
y_train.shape   # (7500,)
y_test.shape    # (2500,)


##############################
## PCA
##############################
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA


## 2D
pca = PCA(n_components = 2)
dx = pca.fit_transform(orig_train)

plt.figure(200)
sns.scatterplot(x= dx[:,0], y = dx[:,1])# hue = y_train.values)

pca = PCA(n_components= 2, whiten= True, random_state = seed)
pca = KernelPCA(n_components = 2, kernel = 'rbf', random_state=seed)

l = train_trans.columns[6:]
l

# Check null status and fill it up with median
#da = org_trans_train[l]
da = train_trans[l]
da.isnull().sum().sum()
da.isnull().sum()[da.isnull().sum() > 0]
nullcols = list(da.isnull().sum()[da.isnull().sum() > 0].index)
nullcols
# Fill up nulls using median
for i in nullcols:
    da[i]= da[i].fillna(da[i].median())

# Check again
da.isnull().sum().sum()
da.columns
da.shape   #  (513, 192)
y_train.shape

dx.shape

from sklearn.feature_selection import VarianceThreshold
sel = VarianceThreshold(threshold=0.006)  # 0.1 indicates 99% of observations approximately
sel.fit(da)
da = sel.transform(da)
da.shape

ss = StandardScaler()
da = pca.fit_transform(ss.fit_transform(da))
da.shape  # (513,2)
plt.figure(300)
sns.scatterplot(x= da[:,0], y = da[:,1], hue = y_train.values)
plt.figure(200)
sns.scatterplot(x= dx[:,0], y = dx[:,1], hue = y_train.values)


colnames = ["c" + str(i) for i in range(dx.shape[1])]
colnames
dx = pd.DataFrame(dx, columns = colnames)
da = pd.DataFrame(da, columns = colnames)




X_train, X_test, ytrain, ytest = train_test_split(dx, y_train, test_size = 0.25 )
Xtrain, Xtest, ytr, yte = train_test_split(da, y_train, test_size = 0.25 )

evals_result= {}
model_pca = xgb.XGBClassifier( n_estimators= 1000,
                           verbosity = 3,
                           eta = 0.06,      # 0.06
                           max_depth = 6,
                           subsample = 0.8,           # 0.8
                           evals_result = evals_result,
                           random_state = 70
                           )


tr_X =  orig_train# Xtrain
test_X = orig_test # Xtest 
ytr = y_train
yte = y_test


model_pca.fit(tr_X, ytr.values,                   # Xtr, ytr
          early_stopping_rounds = 100,   # 10% of n_estimators
          eval_set=[ (test_X, yte.values)],
          eval_metric = ['auc']    # binary classification problem
          )



# auc: 0.81646
model_pca.best_score   # 0.991
pred = model_pca.predict(test_X)
(pred == yte).sum()/yte.size    # 0.937


utils.xg_impt_features(model_pca, tr_X.columns )


##############################
## tsne
##############################
# Why blobs do not appear together in tsne?
# See StackOverflow:
#    https://stats.stackexchange.com/a/453106/78454


from sklearn.manifold import  TSNE


## 2D
tsne = TSNE()
dx = tsne.fit_transform(orig_train)
y_train.values.shape


sns.scatterplot(x= dx[:,0], y = dx[:,1], hue = y_train.values)

tsne = TSNE()
org_trans_train.columns[20:]
da = tsne.fit_transform(org_trans_train[org_trans_train.columns[20:]])
da.shape
sns.scatterplot(x= da[:,0], y = da[:,1], hue = y_train.values)
sns.scatterplot(x= dx[:,0], y = dx[:,1], hue = y_train.values)


## 3D
tsne = TSNE(n_components = 3, early_exaggeration = 40)
dx3 = tsne.fit_transform(orig_train)
dx3.shape


tsne = TSNE(n_components=3)
org_trans_train.columns[20:]
da3 = tsne.fit_transform(org_trans_train[org_trans_train.columns[20:]])
da3.shape

colnames = ["c" + str(i) for i in range(dx3.shape[1])]
colnames
dx3 = pd.DataFrame(dx3, columns = colnames)
da3 = pd.DataFrame(da3, columns = colnames)

dx3['target'] = y_train
da3['target'] = y_train
dx3.head()
da3.head()

os.chdir(master)
dx3.to_csv("dx3.csv", index = False)
da3.to_csv("da3.csv", index = False)




X_train, X_test, ytrain, ytest = train_test_split(dx3.iloc[:,:3], y_train, test_size = 0.25 )
Xtrain, Xtest, ytr, yte = train_test_split(da3.iloc[:,:3], y_train, test_size = 0.25 )

evals_result= {}
model_tsne = xgb.XGBClassifier( n_estimators= 1000,
                           verbosity = 3,
                           eta = 0.06,      # 0.06
                           max_depth = 6,
                           subsample = 0.8,           # 0.8
                           evals_result = evals_result,
                           random_state = 70
                           )


tr_X =  X_train
test_X = X_test 



model_tsne.fit(tr_X, ytrain.values,                   # Xtr, ytr
          early_stopping_rounds = 100,   # 10% of n_estimators
          eval_set=[ (test_X, ytest.values)],
          eval_metric = ['auc']
          )



# auc: 0.81646
model_tsne.best_score   # 1.096898
pred = model_tsne.predict(test_X)
(pred == yte).sum()/yte.size    # 0.75



##############################
## umap
##############################

## 2D

reducer = umap.UMAP()
ss = StandardScaler()
dx = reducer.fit_transform(ss.fit_transform(orig_train))

sns.scatterplot(x= dx[:,0], y = dx[:,1], hue = y_train.values)

reducer = umap.UMAP()
ss = StandardScaler()
org_trans_train.columns[20:]
da = reducer.fit_transform(ss.fit_transform(org_trans_train[org_trans_train.columns[20:]]))
da.shape
sns.scatterplot(x= da[:,0], y = da[:,1], hue = y_train.values)
sns.scatterplot(x= dx[:,0], y = dx[:,1], hue = y_train.values)


colnames = ["c" + str(i) for i in range(dx.shape[1])]
colnames
dx = pd.DataFrame(dx, columns = colnames)
da = pd.DataFrame(da, columns = colnames)




X_train, X_test, ytrain, ytest = train_test_split(dx, y_train, test_size = 0.25 )
Xtrain, Xtest, ytr, yte = train_test_split(da, y_train, test_size = 0.25 )

evals_result= {}
model_umap = xgb.XGBClassifier( n_estimators= 1000,
                           verbosity = 3,
                           eta = 0.06,      # 0.06
                           max_depth = 6,
                           subsample = 0.8,           # 0.8
                           evals_result = evals_result,
                           random_state = 70
                           )


tr_X =  Xtrain
test_X = Xtest 



model_umap.fit(tr_X, ytr.values,                   
          early_stopping_rounds = 100,   # 10% of n_estimators
          eval_set=[ (test_X, yte.values)],
          eval_metric = ['auc']
          )




model_umap.best_score   
pred = model_pca.predict(test_X)
(pred == yte).sum()/yte.size    




#########################################
## Predictive analytics
########################################
# Call it only once
# See https://scikit-learn.org/stable/common_pitfalls.html#general-recommendations


model0 = 0
gc.collect()
del model0
evals_result= {}
model0 = xgb.XGBClassifier( n_estimators= 1000,
                           verbosity = 3,
                           eta = 0.06,      # 0.06
                           max_depth = 6,
                           subsample = 0.8,           # 0.8
                           evals_result = evals_result,
                           random_state = seed
                           )


tr_X =  org_trans_train
test_X =  org_trans_test



model0.fit(tr_X, y_train.values,                   # Xtr, ytr
          early_stopping_rounds = 100,   # 10% of n_estimators
          eval_set=[ (test_X, y_test.values)],
          eval_metric = ['auc']
          )



# auc: 0.81646
model0.best_score   # 0.81761; 820858; 0.816837; 0.892089; 0.876738; 0.884359; 0.885373
                    # 0.84595; 0.851114
pred = model0.predict(test_X)
(pred == y_test).sum()/y_test.size    # 0.7324 0.8022; 0.78395; 0.7954
                                      # 0.7664;0.7716
#plot_importance(model, importance_type = 'gain')



fe_1, fe_0 = xg_impt_features(model0,org_trans_train.columns  )

len(fe_1)   # 335  86  55 76   77  88
len(fe_0)   # 743  11  11 14   16  16



os.chdir(master)
file = open('fe_1.txt','w')
for  item in fe_1:
	file.write(item+"\n")
file.close()

# Read fe_1
os.chdir(master)
with open("fe_1.txt", 'r') as f:
    fe_1 = [line.rstrip('\n') for line in f]
    
len(fe_1)  # 77  88




##---------------
# With reduced best features
model1 = 0
gc.collect()
del model1
evals_result= {}
model1 = xgb.XGBClassifier( n_estimators= 1000,
                           verbosity = 3,
                           eta = 0.06,      # 0.06
                           max_depth = 6,
                           subsample = 0.8,           # 0.8
                           evals_result = evals_result,
                           random_state = seed
                           )


tr_X =  org_trans_train[fe_1[:15]]     # Try from 7 to 30
test_X =  org_trans_test[fe_1[:15]]



model1.fit(tr_X, y_train.values,                   # Xtr, ytr
          early_stopping_rounds = 100,   # 10% of n_estimators
          eval_set=[ (test_X, y_test.values)],
          eval_metric = ['auc']
          )


# auc: 0.81646
model1.best_score   # 0.7228

pred = model1.predict(test_X)
(pred == y_test).sum()/y_test.size    # 0.5244


fe_1[:6]


fe_1[:7]

##--------------------
# orig + binned
##--------------------
gc.collect()
#del model
evals_result= {}
model2 = xgb.XGBClassifier( n_estimators= 1000,
                           verbosity = 3,
                           eta = 0.06,      # 0.06
                           max_depth = 6,
                           subsample = 0.8,           # 0.8
                           evals_result = evals_result,
                           random_state = seed
                           )


tr_X =  org_binned_train
test_X =  org_binned_test



model2.fit(tr_X, y_train.values,                   # Xtr, ytr
          early_stopping_rounds = 50,   # 10% of n_estimators
          eval_set=[ (test_X, y_test.values)],
          eval_metric = ['auc']
          )



# auc: 0.81646
model2.best_score   # 0.821435 ; 827361 ; 0.897
pred = model2.predict(test_X)
(pred == y_test).sum()/y_test.size    # 0.7324 ; 0.81

fe_11, fe_00 = xg_impt_features(model2,org_binned_train.columns  )
len(fe_11)
fe_00

##-------------------
# orig + binned best features
##-------------------


gc.collect()
#del model
evals_result= {}
model3 = xgb.XGBClassifier( n_estimators= 1000,
                           verbosity = 3,
                           eta = 0.06,      # 0.06
                           max_depth = 6,
                           subsample = 0.8,           # 0.8
                           evals_result = evals_result,
                           random_state = rng
                           )


tr_X =  org_binned_train[fe_11]
test_X =  org_binned_test[fe_11]



model3.fit(tr_X, y_train.values,                   # Xtr, ytr
          early_stopping_rounds = 50,   # 10% of n_estimators
          eval_set=[ (test_X, y_test.values)],
          eval_metric = ['auc']
          )



# auc: 0.81646
model3.best_score   # 826236; 826423
pred = model3.predict(test_X)
(pred == y_test).sum()/y_test.size    # 0.7324



##--------------------
##-------------------
# orig  features
##-------------------


model4 = 0

gc.collect()
del model4
evals_result= {}
model4 = xgb.XGBClassifier( n_estimators= 1000,
                           verbosity = 3,
                           eta = 0.06,      # 0.06
                           max_depth = 6,
                           subsample = 0.8,           # 0.8
                           evals_result = evals_result,
                           random_state = seed
                           )


tr_X =  orig_train[fe_4_1[:5]]
test_X =  orig_test[fe_4_1[:5]]



model4.fit(tr_X, y_train.values,                   # Xtr, ytr
          early_stopping_rounds = 100,   # 10% of n_estimators
          eval_set=[ (test_X, y_test.values)],
          eval_metric = ['auc']
          )



# auc: 0.81646
model4.best_score   # 0.7335065739582236
pred = model4.predict(test_X)
(pred == y_test).sum()/y_test.size    # 0.544

fe_4_1, fe_4_0 = xg_impt_features(model4,orig_train.columns  )

fe_4_1[:5]

##--------------------

fe_4_1[:5]

model4_1 = 0

gc.collect()
del model4_1
evals_result= {}
model4_1 = xgb.XGBClassifier( n_estimators= 1000,
                           verbosity = 3,
                           eta = 0.06,      # 0.06
                           max_depth = 6,
                           subsample = 0.8,           # 0.8
                           evals_result = evals_result,
                           random_state = 70
                           )


tr_X =  orig_train[fe_4_1[:5]]
test_X =  orig_test[fe_4_1[:5]]



model4_1.fit(tr_X, y_train.values,                   # Xtr, ytr
          early_stopping_rounds = 100,   # 10% of n_estimators
          eval_set=[ (test_X, y_test.values)],
          eval_metric = ['auc']
          )



# auc: 0.81646
model4_1.best_score   # 831523 ; 824436 ; 0.8288 ; 0.897301 ; 0.880147; (0.891444, 0.892768, 0.893049)
                    # (0.858484,0.862771, 0.874083 )
pred = model4_1.predict(test_X)
(pred == y_test).sum()/y_test.size    # 0.7376 ; 0.81; 0.7881; 0.8014, 0.8044
                                      # 0.7788; 0.7918

###################################




###################################
####################################


y = train_train.pop('target')
train_train.head()
ohe = OneHotEncoder(  sparse = False)
ohe.fit(train_train)
train_ohe = ohe.transform(train_train)
train_ohe.shape  # (7500, 89)
cl = ["c" + str(i) for i in range(train_ohe.shape[1]) ]
train_ohe = pd.DataFrame(train_ohe,columns = cl)
train_ohe.head()
train_ohe.shape  # (7500,75)




pca = PCA(n_components=3)
train_pca= pca.fit_transform(train_ohe)
train_ohe.head()
cx = ["c" + str(i) for i in range(train_pca.shape[1]) ]
train_pca = pd.DataFrame(train_pca,columns = cx)
train_pca.head()



os.chdir(dataPath)

train_pca.to_csv("train_pca.csv", index = False)
y.to_csv("y_train_pca.csv", index = False)
y.head()


##################Model with orig data #####################


X = orig_train  
y = orig_train.pop('target')
X.columns
X.head()
y    

X_train,X_test,y_train,y_test = train_test_split( X,y,
                                                 test_size = 0.25,
                                                 stratify = y,
                                                 random_state = 384)

gc.collect()
#del model
evals_result= {}
model = xgb.XGBClassifier( n_estimators= 700,
                           verbosity = 3,
                           eta = 0.06,      # 0.06
                           max_depth = 6,
                           subsample = 0.8,           # 0.8
                           evals_result = evals_result,
                           random_state = 800
                           )

tr_X =  X_train
test_X =  X_test


model.fit(tr_X, y_train,                   # Xtr, ytr
          early_stopping_rounds = 50,   # 10% of n_estimators
          eval_set=[ (test_X, y_test)],
          eval_metric = ['merror']
          )



pred = model.predict(test_X)
(pred == y_test).sum()/y_test.size    # 94.93%   91.8%  94.73  98.2(class_Sep = 2.0)
plot_importance(model, importance_type = 'gain')

################## Model with discrete features #####################


X = train_train  
y = train_train.pop('target')
X.columns
X.head()
y    

for i,j in enumerate(X.columns):
    X[j] = X[j].astype('int') 


X_train,X_test,y_train,y_test = train_test_split( X,y,
                                                 test_size = 0.25,
                                                 stratify = y,
                                                 random_state = 384)

gc.collect()
del model
evals_result= {}
model = xgb.XGBClassifier( n_estimators= 700,
                           verbosity = 3,
                           eta = 0.06,      # 0.06
                           max_depth = 6,
                           subsample = 0.8,           # 0.8
                           evals_result = evals_result,
                           random_state = 800
                           )

tr_X =  X_train
test_X =  X_test


model.fit(tr_X, y_train,                   # Xtr, ytr
          early_stopping_rounds = 50,   # 10% of n_estimators
          eval_set=[ (test_X, y_test)],
          eval_metric = ['merror']
          )



pred = model.predict(test_X)
(pred == y_test).sum()/y_test.size    # 94.6% ; 95%  90.8%  94.86  98.86(class sep = 2.0)
plot_importance(model, importance_type = 'gain')

##############################################################



import matplotlib.pyplot as plt
import seaborn as sns
fig,ax= plt.subplots(1,1,figsize = (10,10))
sns.scatterplot(data = tr_X, x = 'fe', y = 'fd', hue= y_train, ax = ax, alpha = 0.4)

fig,ax= plt.subplots(1,1,figsize = (10,10))
sns.scatterplot(data = orig_train, x = 'fe', y = 'fb', hue= y,ax=ax ,palette = "Set2")





#################################################################



plt.figure(1)
plt.clf()
colors = ["#dede00", "#377eb8", "#f781bf"]
markers = ["x", "o", "^"]

# Three clusters can be seen
fig = plt.figure(figsize = (8,8))
_=sns.scatterplot(data = X, x = "x1", y = "x2", hue = y)

fig = plt.figure(figsize = (8,8))
_=sns.scatterplot(data = X, x = "x2", y = "x3", hue = y)


fig = plt.figure(figsize = (8,8)) ;
_=sns.scatterplot(data = X, x = "x1", y = "x3", hue = y)
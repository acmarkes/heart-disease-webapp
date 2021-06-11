#%%
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from random import randint
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2, f_classif

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error

from sklearn.model_selection import KFold, train_test_split, cross_val_score, GridSearchCV


#%%
col_names = ['age', 'sex',
             'cp','trestbps',
             'chol','fbs',
             'restecg','thalach','exang',
             'oldpeak','slope',
             'ca','thal','num']
""" Attr explanation
      -- 1. #3  (age)       
      -- 2. #4  (sex)       
      -- 3. #9  (cp) : chest pain type
						-- Value 1: typical angina
						-- Value 2: atypical angina
						-- Value 3: non-anginal pain
						-- Value 4: asymptomatic
      -- 4. #10 (trestbps) : resting blood pressure (in mm Hg on admission to the hospital) 
      -- 5. #12 (chol): serum cholestoral in mg/dl      
      -- 6. #16 (fbs): (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)       
      -- 7. #19 (restecg): resting electrocardiographic results   
						-- Value 0: normal
						-- Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)
						-- Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria
      -- 8. #32 (thalach): maximum heart rate achieved   
      -- 9. #38 (exang): exercise induced angina (1 = yes; 0 = no)     
      -- 10. #40 (oldpeak):ST depression induced by exercise relative to rest   
      -- 11. #41 (slope): the slope of the peak exercise ST segment     
      -- 12. #44 (ca): number of major vessels (0-3) colored by flourosopy        
      -- 13. #51 (thal): 3 = normal; 6 = fixed defect; 7 = reversable defect      
      -- 14. #58 (num): diagnosis of heart disease (angiographic disease status) TARGET
						-- Value 0: < 50% diameter narrowing
						-- Value 1: > 50% diameter narrowing
"""

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/'

cleve_url = f'{url}processed.cleveland.data'

cleve_df = pd.read_csv(cleve_url,index_col = False, names=col_names)
cleve_df.columns.name = 'Cleveland'

#%%
#QUICK DF LOOKUP
def peek(frame):
    print(f"Here's a summary of the {frame.columns.name} dataset with dimensions {frame.shape}")
    print(frame.head())
    print("-"*80)
    print("Its columns are:")
    print(frame.info())
    print("-"*80)
    print("Here are its statistical characteristics:")
    print(frame.describe(include='all'))
    #frame.groupby('num').size()

#%%
def feature_scaling(method, frame, columns=None):

    if method == 'std':
        transformer = Pipeline(steps=[('standard', StandardScaler())])
    if method == 'mn':
        transformer = Pipeline(steps=[('minmax', MinMaxScaler())])
    if columns == None:
        columns = list(frame.columns)


    preprocessor = ColumnTransformer(
            remainder='drop', #passthough features not listed
            transformers=[
                #('std', standard_transformer , ['thalach','oldpeak','chol']),
                ('trnsf', transformer , columns)
            ])
    
    processed_frame = pd.DataFrame(preprocessor.fit_transform(frame)) 
    processed_frame.columns = columns    
    processed_frame.dropna(inplace=True)   
    return processed_frame


#%%
#CLEANING UP THE FRAMES
def preprocessing(frame):
    try:
        frame['thal'].replace({"?":np.nan}, inplace=True)
        frame['ca'].replace({"?":np.nan}, inplace=True)
        frame.dropna(subset = ['cp','ca','thal'],inplace=True)
        frame[['cp','ca','thal']] = frame[['cp','ca','thal']].astype('category')
        frame.loc[frame['num'] > 1, 'num'] = 1
    except:
        pass
    frame = pd.get_dummies(frame)

    cols = list(frame.columns.values) #Make a list of all of the columns in the df
    cols.pop(cols.index('num')) #Remove b from list
    frame = frame[cols+['num']] #Create new dataframe with columns in the wanted order


    frame.dropna(inplace=True)

    return frame

#%%
#CLASSIFICATION FNCTION WITH CROSSVAL BUILT-IN
def classification_model(model, data, predictors, outcome, folds = 5, seed = randint(0, 5000)):
    #Fit the model
    model.fit(data[predictors],data[outcome])
    
    #Make predictions on training set
    predictions = model.predict(data[predictors])

    #Print accuracy
    accuracy = accuracy_score(predictions,data[outcome])
    print(f"Accuracy: {accuracy*100:.4f}% (0:.3%)")

    #Perform k-fold cross-validation with 5 folds
    kf = KFold(n_splits = folds, shuffle = True, random_state = seed)
    error = []
    for train,test in kf.split(data):
        #Filter training data
        train_predictors = (data[predictors].iloc[train,:])

        #Target to train the algorithm
        train_target = data[outcome].iloc[train]

        #Training the algorithm using the predictors and target
        model.fit(train_predictors,train_target)

        #Record error from each cross-validation run
        error.append(model.score(data[predictors].iloc[test,:],data[outcome].iloc[test]))
        print(f"Cross-Validation Score: {np.mean(error)*100:.4f}% ( 0:.3% )")

        #Fit the model again so it can be referred outside the function
        model.fit(data[predictors],data[outcome])


#%%
def feature_ranking(frame,k=5,print=False):
    '''
    Function for feature, giving Kbest and k-th most important features according to RandomForest
    frame = dataframe to be evaluated
    k = number of features
    print = print out the sorted dictionary of features and their scores and importances

    Function return a dict with the structure type: list of kth features. Capture them by choosing feature_ranking['scores'] or feature_ranking['importances']
    
    '''
    #KBest set up
    array = frame.values
    X = array[:,0:-1]
    Y = array[:,-1]
    # feature extraction
    kbest = SelectKBest(score_func=f_classif, k=k)
    extracted = kbest.fit(X, Y)
    # summarize scores
    np.set_printoptions(precision=3)
    #print(fit.scores_)
    features = extracted.transform(X)
    scores = list(extracted.scores_)
    cols = list(frame.columns)
    named_features = dict(zip(cols,scores))
    sorted_scores = sorted(named_features.items(), key = lambda kv: kv[1], reverse= True)
   
    model = RandomForestClassifier(n_estimators=25, min_samples_split=25,max_depth=7,max_features=1)
    model.fit(X,Y)
    importances = dict(zip(list(frame.columns),model.feature_importances_) )
    sorted_importances = sorted(importances.items(), key = lambda kv: kv[1], reverse= True)
    
    if print:
        print(sorted_scores)
        print(sorted_importances)
    
    
    
    return {'scores':list(dict(sorted_scores[:k]).keys()), 'importances':list(dict(sorted_importances[:k]).keys())}


#%%
cleve_df = preprocessing(cleve_df)
columns_to_scale = ['thalach', 'chol','oldpeak']
scaled_columns = feature_scaling('std',cleve_df,columns_to_scale)

for column in columns_to_scale:
    cleve_df[column] = scaled_columns[column]
cleve_df.dropna(inplace=True)
print(feature_ranking(cleve_df)['importances'])

#%%
data = cleve_df

#models
logis = LogisticRegression(max_iter=800)
rndf = RandomForestClassifier()
supv = SVC()
models = [logis,rndf,supv]

#predictors
corrs = ['thal_3.0','cp_4.0','ca_0.0','thal_7.0','exang','slope']
scores = feature_ranking(cleve_df,5)['scores']
importances = feature_ranking(cleve_df,5)['importances']
predictors = [corrs,scores,importances]

#target
outcome = 'num'

#%%
for model in models:
    for p in predictors:
        print(model)
        print(p)
        print('-'*50)
        classification_model(model,data,p,outcome,seed = 132)
        print('-'*50)


#Best model = supv
#Best predictors = corrs

#%%
#Tuning SVC hyperparameters
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 2, 3], 'probability':[True]}
clf = GridSearchCV(supv, parameters)
clf.fit(data.iloc[:,0:-1], data.iloc[:,-1])
supv = clf.best_estimator_

# %%
X = data.iloc[:,0:-1]
Y = data.iloc[:,-1]

test_size = 0.15
seed = 17

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size,
random_state=seed)
model = supv.fit(X_train[corrs], Y_train)

#%%
result = model.score(X_test[corrs], Y_test)
print(result*100.0)


# %%
num_folds = 10
seed = 7
kfold = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean()*100.0, results.std()*100.0)

# %%
model = supv.fit(X[corrs], Y)

import pickle
pickle.dump(model, open('heartd_clf.pkl', 'wb'))
# %%
['thal_3.0', 'cp_4.0', 'ca_0.0', 'thal_7.0', 'exang', 'slope']

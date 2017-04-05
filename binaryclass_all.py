import csv
import pandas as pd
import numpy as np
import sys

from zipfile import ZipFile
from datetime import datetime
import time

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import MultinomialNB

from fancyimpute import MICE, BiScaler, KNN, NuclearNormMinimization, SoftImpute

from sklearn.linear_model import RandomizedLasso

from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif, SelectPercentile

debug = False

def drop_low_var_quality_features(df, missing_values_threshold, std_threshold):
    """
    Params:
    missing_values_threshold: Maximum number of missing values for a column to be retained
    std_threshold: Minimum standard deviation for a column to be retained.
    """
    rows_count, cols_count = df.shape
    df.dropna(axis=1, thresh=rows_count-missing_values_threshold, inplace=True)
    df.drop(df.std()[df.std() < std_threshold].index.values, axis=1, inplace=True)

def is_valid_type(values):
    validity_list = []
    for val in values:
        isValid = True if type(val) == bool or np.isfinite(val) else False
        validity_list.append(isValid)
    return validity_list
    
def get_data_for_characteristic(X_train, Y_train, characteristic, get_only_complete_cases=False):
    y_char = Y_train[is_valid_type(Y_train[characteristic])]
    
    training_ids = y_char.index.values.tolist()
    X_char = X_train.ix[training_ids]
    
    non_numeric_cols = X_char.select_dtypes(exclude=[np.number]).columns.values.tolist()
    X_char.drop(non_numeric_cols, axis=1, inplace=True)
    
    if get_only_complete_cases is True:
        X_char = X_char.dropna(axis=0, inplace=False)
        training_ids = X_char.index.values.tolist()
        y_char = y_char.ix[training_ids]

    assert(y_char.index.values.tolist() == X_char.index.values.tolist())
    return X_char, y_char[characteristic]


def perform_imputation_for_characteristic(df, y_train_df, selection_threshold, characteristic, imputeKNN=False):
    # Get Xs and corresponding Ys for a characteristic. Only get the complete Xs (without any nan values)
    X_char_complete, y_char_complete = get_data_for_characteristic(df, y_train_df, characteristic, get_only_complete_cases=True)
    
    # Convert the complete cases DFs to Matrix
    X_char_c_mat, y_char_c_mat = X_char_complete.as_matrix(), y_char_complete.as_matrix()
    
    # Perform Randomized Lasso feature selection using the complete cases.
    lasso = RandomizedLasso(alpha='aic', random_state=39, n_resampling=500)
    lasso.fit(X_char_c_mat, y_char_c_mat)
    stability_scores = lasso.scores_
    support = np.where(stability_scores > selection_threshold)[0]
    
    print selection_threshold, '\t', str(len(support))
    if debug:
        print str(support)
    
    # Shrink the the X's dataframe to only required features
    XFFull = df.iloc[:,support]
    
    # Perform the imputation
    XFFull_mat = XFFull.as_matrix()
    try:
        if imputeKNN:
            X_full_imputed = KNN(k=3).complete(XFFull_mat)
        else:
            X_full_imputed = MICE().complete(XFFull_mat)
    except ValueError:
        X_full_imputed = XFFull_mat
        print "Input matrix is not missing any values."
    
    print "NaN's before imputation {}, after {}.".format(np.count_nonzero(np.isnan(XFFull_mat)), np.count_nonzero(np.isnan(X_full_imputed)))
    print "NaN was at {}".format(np.argwhere(np.isnan(XFFull_mat)))
    
    return pd.DataFrame(X_full_imputed, index=XFFull.index, columns=XFFull.columns)


def run_for_nomial_char(df, y_train_df, characteristic, useAdaBoost=False):
    imputed_pd = perform_imputation_for_characteristic(df, y_train_df, .00001, characteristic)
    if useAdaBoost:
        tuned_parameters = {'n_estimators': [50, 100, 300], 'learning_rate': [0.5, 1, 2] }
        clf = GridSearchCV(AdaBoostClassifier(), tuned_parameters, cv=3, scoring='roc_auc', verbose=5, n_jobs=-1)
    else:
        clf = LogisticRegressionCV(cv=3, scoring='roc_auc', verbose=10, n_jobs=-1)
    X_train, y_train = get_data_for_characteristic(imputed_pd, y_train_df, characteristic)
    clf = clf.fit(X_train.as_matrix(), y_train.as_matrix().astype(int))
    predictions = clf.predict(imputed_pd)
    print "predicted {} as positive out of {}.".format(sum(predictions), len(predictions))
    return predictions

def gen_submission(pred, name=""):
    pred_filename = "prediction" + name + ".csv" 
    pred.to_csv(pred_filename, index=False)
    with ZipFile( str('Submission' + name + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '.zip'), 'w') as myzip:
        myzip.write(pred_filename, arcname='prediction.csv')
        myzip.write('narrative.txt')
        myzip.write('ffc.py')

def run_for_nomial_char_new(df, y_train_df, characteristic, method):
    x_complete, y_complete = get_data_for_characteristic(df, y_train_df, characteristic, get_only_complete_cases=True)
    assert (x_complete.index.values.tolist() == y_complete.index.values.tolist())
    x_complete_mat, y_complete_list = x_complete.as_matrix(), y_complete.tolist()
    full_mat = df.as_matrix()
    len(df.index.values.tolist())
    
    # Prelimnary feature selection
    selector = SelectKBest(mutual_info_classif, k=100)
    train_X_sel = selector.fit_transform(x_complete_mat, y_complete_list)
    
    # perform impute
    transformed_full_mat = full_mat[:, selector.get_support(True)]

    try:
        X_full_imputed = MICE().complete(transformed_full_mat)
    except ValueError:
        X_full_imputed = transformed_full_mat
        print "Input matrix is not missing any values."

    print "NaN's before imputation {}, after {}.".format(np.count_nonzero(np.isnan(transformed_full_mat)), np.count_nonzero(np.isnan(X_full_imputed)))
    print "NaN was at {}".format(np.argwhere(np.isnan(X_full_imputed)))
    
    imputed_pd = X_full_imputed
   
    train_idxs = y_train_df.index.values.tolist()
    train_idxs[:] = [x - 1 for x in train_idxs]
    
    y_train = np.array(y_train_df[characteristic].tolist()).astype(int)
    X_train = imputed_pd[train_idxs, :]
    
    if method == 0:
        tuned_parameters = {'n_estimators': [50, 100, 300], 'learning_rate': [0.5, 1, 2] }
        clf = GridSearchCV(AdaBoostClassifier(), tuned_parameters, cv=3, scoring='roc_auc', verbose=5, n_jobs=-1)
    elif method == 2:
        clf = DecisionTreeClassifier()
    else:
        clf = MultinomialNB()
    
    clf = clf.fit(X_train, y_train.astype(int))
    predictions = clf.predict(imputed_pd)
    print "predicted {} as positive out of {}.".format(sum(predictions), len(predictions))
    return predictions


df = pd.read_csv('background.csv', low_memory=False)
df.cf4fint = ((pd.to_datetime(df.cf4fint) - pd.to_datetime('1960-01-01')) / np.timedelta64(1, 'D')).astype(int)

non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.values.tolist()
df.drop(non_numeric_cols, axis=1, inplace=True)

df.index = df['challengeID']
df.sort_index(inplace=True)

if debug:
    # number of nan cols
    print sum(df.isnull().sum().tolist())
    # number of non-nan cols
    print df.count().sum()
    # Total elements
    print df.size
    # Shape
    print df.shape

drop_low_var_quality_features(df, 200, 0.2)

if debug:
    print sorted(df.isnull().sum().tolist(), reverse=True)[:10]
    print sorted(df.std())[:30]
    print df.shape

y_train_df = pd.read_csv("train.csv", low_memory=False)
y_train_df.index = y_train_df['challengeID']
y_train_df.sort_index(inplace=True)

prediction = pd.read_csv("prediction_old.csv", low_memory=False)

lr_prediction = prediction.copy(deep=True)
ab_prediction = prediction.copy(deep=True)
dt_prediction = prediction.copy(deep=True)
mnb_prediction = prediction.copy(deep=True)

for characteristic in ['eviction' ,'layoff' ,'jobTraining']:
    print characteristic
    ab_prediction[characteristic] = run_for_nomial_char_new(df, y_train_df, characteristic, 0)
    dt_prediction[characteristic] = run_for_nomial_char_new(df, y_train_df, characteristic, 2)
    mnb_prediction[characteristic] = run_for_nomial_char_new(df, y_train_df, characteristic, 3)


gen_submission(ab_prediction,"_ab")
gen_submission(dt_prediction,"_dt")
gen_submission(mnb_prediction,"_mnb")
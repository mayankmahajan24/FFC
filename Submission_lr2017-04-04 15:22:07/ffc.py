import csv
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy import linalg
import statsmodels.regression.linear_model as lm
from sklearn import datasets
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold, KFold, GridSearchCV
from sklearn.linear_model import LassoCV, LassoLarsCV, LassoLarsIC, RandomizedLasso, lasso_stability_path, ElasticNet, Lasso, Ridge
from sklearn.feature_selection import f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import auc, precision_recall_curve, mean_squared_error
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.utils.extmath import pinvh
from sklearn.exceptions import ConvergenceWarning
import warnings
from zipfile import ZipFile
from datetime import datetime
import time

supports = {}
thresholds = np.logspace(-3,-1,3)
clf_names = ['OLS','LASSO', 'Ridge','ElasticNet']
results = pd.DataFrame(index=thresholds, columns=clf_names)
alphas = pd.DataFrame(index=thresholds, columns=clf_names[1:])
plt.interactive(True)
best_thresholds = {}

'''
    Filtering
        remove object columns
        remove numeric columns that are all NA
    Imputation
        use median
    Filter by response variable not NA

    Feature Selection:
    IC selection breaks when matrix is poorly conditioned.
    LASSO arbitrarily selects features 
    Select alpha through stability selection, bootstrapping
    RandomizedLasso [B2009 https://hal.inria.fr/hal-00354771/]
                    [M2010 https://arxiv.org/pdf/0809.2932.pdf]

    Classifiers:
        LineReg
        LASSO
        Ridge
        Elastic

'''

def fillMissing(df):    
    # read input csv - takes time
    #df = pd.read_csv(inputcsv, low_memory=False)

    # replace NA's with median
    med2 = df.median()
    dfi = df.fillna(value=med2)
    
    #remove object columns
    dfn= dfi.select_dtypes(['number'])

    #remove NA columns
    a = dfn.notnull().all()
    dfn = dfn[a[a].index]

    # replace negative values with 1
    num = dfn._get_numeric_data()
    num[num < 0] = 1
    # write filled outputcsv
    return dfn
    #dfn.to_csv(outputcsv, index=False)

def drop_low_var_quality_features(df, missing_values_threshold, std_threshold):
    """
    Params:
    missing_values_threshold: Maximum number of missing values for a column to be retained
    std_threshold: Minimum standard deviation for a column to be retained.
    """
    rows_count, cols_count = df.shape
    df.dropna(axis=1, thresh=rows_count-missing_values_threshold, inplace=True)
    df.drop(df.std()[df.std() < std_threshold].index.values, axis=1, inplace=True)

def filter_data(df):
    nRow = len(df) 
    nCol = len(df.iloc[0,:])

    #Drop low quality features
    rows_count, cols_count = df.shape
    df.dropna(axis=1, thresh=rows_count-200, inplace=True)
    df.drop(df.std()[df.std() < 0.2].index.values, axis=1, inplace=True)
        
    #Drop nonnumeric columns
    df.index = df['challengeID']
    df = df.drop(['challengeID', 'idnum'], axis=1)
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.values.tolist()
    df.drop(non_numeric_cols, axis=1, inplace=True)

    #Sort by challenge ID / index
    df.sort_index(inplace = True)
    return df

def imputation(X_all, characteristic): #Takes post early removal X

    #Imputation
    X_imputed = fillMissing(X_all)

    #Get training challenge IDs
    Y_train = pd.read_csv("train.csv", low_memory=False)
    Y_train.index = Y_train['challengeID']
    Y_train.sort_index(inplace=True)

    return X_imputed, Y_train #Note, different sizes still

def get_data_for_characteristic(X_train, Y_train, characteristic, get_only_complete_cases=False):
    y_char = Y_train[np.isfinite(Y_train[characteristic])]
    
    training_index = y_char.index
    X_char = X_train.loc[training_index,:]
    #assert(training_ids == X_char['challengeID'].tolist())
        
    return X_char, y_char[characteristic]

def feature_selection(X, y):
    print "Feature Selection"
    X = X.as_matrix()
    y = y.as_matrix()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        warnings.simplefilter('ignore', ConvergenceWarning)
        lasso = RandomizedLasso(alpha='aic', random_state=39, n_resampling=500)
        lasso.fit(X,y)      
        #plot_stability_path()

    return lasso

def plot_stability_path():
    plt.figure()
    # We plot the path as a function of alpha/alpha_max to the power 1/3: the
    # power 1/3 scales the path less brutally than the log, and enables to
    # see the progression along the path
    print "\tLasso Stability Path"
    alpha_grid, scores_path = lasso_stability_path(X, y, random_state=43, eps=0.05)

    hg = plt.plot(alpha_grid[1:] ** .333, scores_path[coef != 0].T[1:], 'r')
    hb = plt.plot(alpha_grid[1:] ** .333, scores_path.T[1:], 'k')
    ymin, ymax = plt.ylim()
    plt.xlabel(r'$(\alpha / \alpha_{max})^{1/3}$')
    plt.ylabel('Stability score: proportion of times selected')
    plt.title('Stability Scores Path')# - Mutual incoherence: %.1f' % mi)
    plt.axis('tight')
    plt.legend((hg[0], hb[0]), ('relevant features', 'irrelevant features'),
               loc='best')
    plt.show()

def gen_grid(X,y):
    global supports, results, alphas, best_thresholds
    supports = {}
    results = pd.DataFrame(index=thresholds, columns=clf_names)
    alphas = pd.DataFrame(index=thresholds, columns=clf_names[1:])
    best_thresholds = {}

    randomized_lasso = feature_selection(X,y)
    stability_scores = randomized_lasso.scores_

    for threshold in thresholds:
        support = np.where(stability_scores > threshold)[0]
        print threshold, '\t', str(support)
        supports[threshold] = support
        Xf = X.iloc[:,support]

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            warnings.simplefilter('ignore', ConvergenceWarning)
            #OLS

            print "\tOLS"
            ols_fit = lm.OLS(y, Xf).fit()
            results.ix[threshold,'OLS'] = ols_fit.mse_resid
            #print "OLS"

            param_grid = dict(alpha=np.logspace(-6,0,7))
            cv = StratifiedKFold(n_splits=5, random_state=42)

            #LASSO
            print "\tLASSO"
            lasso_grid = GridSearchCV(Lasso(), param_grid=param_grid, cv=cv, scoring='neg_mean_squared_error')
            lasso_grid.fit(Xf,y)
            results.ix[threshold,'LASSO'] = lasso_grid.best_score_
            alphas.ix[threshold,'LASSO'] = lasso_grid.best_params_['alpha']
            
            #Ridge
            print "\tRidge"
            ridge_grid = GridSearchCV(Ridge(), param_grid=param_grid, cv=cv, scoring='neg_mean_squared_error')
            ridge_grid.fit(Xf,y)
            results.ix[threshold,'Ridge'] = ridge_grid.best_score_
            alphas.ix[threshold,'Ridge'] = ridge_grid.best_params_['alpha']
            
            #Elastic
            print "\tElasticNet"
            elastic_grid = GridSearchCV(ElasticNet(), param_grid=param_grid, cv=cv, scoring='neg_mean_squared_error')
            elastic_grid.fit(Xf,y)
            results.ix[threshold,'ElasticNet'] = elastic_grid.best_score_
            alphas.ix[threshold,'ElasticNet'] = elastic_grid.best_params_['alpha']
    results = results.abs()

    for name in clf_names:
        best_thresholds[name] = results[name].idxmin()
    return results, alphas

def make_threshold_plot():
    plt.figure()

    plt.xscale('log')

    ols_plot = plt.plot(thresholds, results['OLS'], 'r')
    lasso_plot = plt.plot(thresholds, results['LASSO'], 'b')
    ridge_plot = plt.plot(thresholds, results['Ridge'], 'o')
    elastic_plot = plt.plot(thresholds, results['ElasticNet'], 'g')
    plt.xlabel('Stability score: proportion of times selected')
    plt.ylabel('Mean squared error on training data')
    plt.axis('tight')
    plt.legend((ols_plot[0], lasso_plot[0], ridge_plot[0], elastic_plot[0]), ('OLS', 'LASSO', 'Ridge', 'ElasticNet'),
               loc='best')
    plt.show()

def generate_all_predictions(X,y,background,characteristic): #Generates predictions from the 4 classifiers for a characteristic
    print "Generating for " + characteristic
    clfs = [
    lm.OLS(y, X.iloc[:,supports[best_thresholds['OLS']]]).fit(),
    Lasso(alpha=alphas.ix[best_thresholds['LASSO'],'LASSO']).fit(X.iloc[:,supports[best_thresholds['LASSO']]],y),
    Ridge(alpha=alphas.ix[best_thresholds['Ridge'],'Ridge']).fit(X.iloc[:,supports[best_thresholds['Ridge']]],y),
    ElasticNet(alpha=alphas.ix[best_thresholds['ElasticNet'],'ElasticNet']).fit(X.iloc[:,supports[best_thresholds['ElasticNet']]],y)
    ]
    predictions = {}
    for (clf,name) in zip(clfs,clf_names):
        predictions[name] = clf.predict(background.iloc[:,supports[best_thresholds[name]]])
    return predictions


def gen_submission(pred, name=""):
    pred_filename = "prediction" + name + ".csv" 
    pred.to_csv(pred_filename, index=False)
    with ZipFile( str('Submission' + name + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '.zip'), 'w') as myzip:
        myzip.write(pred_filename, arcname='prediction.csv')
        myzip.write('narrative.txt')
        myzip.write('ffc.py')


def main2():
    # print "Read"
    # background = pd.read_csv("background.csv", low_memory=False)
    # # Fix date bug
    # background.cf4fint = ((pd.to_datetime(background.cf4fint) - pd.to_datetime('1960-01-01')) / np.timedelta64(1, 'D')).astype(int)

    # background.sort_values(by='challengeID', inplace=True)
    # background.index = background['challengeID']
    
    # #Removal of bad features
    # print "Initial Filter"
    # X_all = filter_data(background)

    # #impute and select data based on characteristic (KEEP FOR MAYANK / DELETE FOR AKASH)
    # print "Imputation (Mayank)"
    # X_imputed, y_imputed = imputation(X_all, None)

    # # one time code, write median imputed code to csv
    # X_imputed.to_csv("X_median_imputed.csv", index=True)

    X_imputed = pd.read_csv("X_median_imputed.csv", low_memory=False, index_col=0)
    X_imputed.cf4fint = ((pd.to_datetime(X_imputed.cf4fint) - pd.to_datetime('1960-01-01')) / np.timedelta64(1, 'D')).astype(int)
    y_imputed = pd.read_csv("train.csv", low_memory=False, index_col=0)

    prediction = pd.read_csv("prediction_old.csv", low_memory=False)

    ols_prediction = prediction.copy(deep=True)
    lasso_prediction = prediction.copy(deep=True)
    ridge_prediction = prediction.copy(deep=True)
    elastic_prediction = prediction.copy(deep=True)

    for characteristic in ['grit', 'gpa', 'materialHardship']:
        print characteristic
        #X_imputed, y_all = imputation(X_all, characteristic)  (KEEP FOR AKASH / DELETE FOR MAYANK)

        X,y = get_data_for_characteristic(X_imputed, y_imputed, characteristic)
        results,alphas = gen_grid(X,y)
        results.to_csv("scores_" + characteristic + ".csv", index=False)
        alphas.to_csv("alphas_" + characteristic + ".csv", index=False)
        #make_threshold_plot()
        predictions = generate_all_predictions(X,y,X_imputed,characteristic)

        ols_prediction[characteristic] = predictions['OLS']
        lasso_prediction[characteristic] = predictions['LASSO']
        ridge_prediction[characteristic] = predictions['Ridge']
        elastic_prediction[characteristic] = predictions['ElasticNet']

    gen_submission(ols_prediction,"_ols")
    time.sleep(2)
    gen_submission(lasso_prediction,"_lasso")
    time.sleep(2)
    gen_submission(ridge_prediction,"_ridge")
    time.sleep(2)
    gen_submission(elastic_prediction,"_elastic")
    time.sleep(2)

'''
def main():
    #Impute data.
    #fillMissing('background.csv', 'output.csv') #Comment this out after one run

    background = pd.read_csv("output.csv", low_memory=False)
    background.sort_values(by='challengeID', inplace=True)
    background.index = background['challengeID']
    prediction = pd.read_csv("prediction_old.csv", low_memory=False)

    X_all,y_all = filter_data(background)
    X,y = get_data_for_characteristic(X_all, y_all, 'grit')

    randomized_lasso = feature_selection(X,y)
    Xf = X.loc[:,randomized_lasso.get_support()]

    #Regular OLS on grit
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        warnings.simplefilter('ignore', ConvergenceWarning)
        olsf = lm.OLS(y, Xf)
        print "OLS"
        print olsf.fit().summary()

    #Prediction
    testf = randomized_lasso.transform(basedckground.drop(['challengeID','idnum'],axis=1))
    grit_predict = olsf.fit().predict(testf)
    grit_predict_round = np.round(grit_predict*4)/4 #Round to nearest 0.25
    grit_predict_round = np.where(grit_predict_round < 4.0, grit_predict_round, 4.0) #Bounds
    grit_predict_round = np.where(grit_predict_round > 1.0, grit_predict_round, 1.0) #Bounds

    prediction['grit'] = grit_predict_round

    print "Training MSE (no rounding): " + str(mean_squared_error(y.as_matrix(), grit_predict[y.index.values]))
    print "Training MSE (w/rounding): " + str(mean_squared_error(y.as_matrix(), prediction.ix[y.index, 'grit'].as_matrix()))

    gen_submission(prediction)
'''
if __name__ == "__main__":
    main2()


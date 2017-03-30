import csv
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy import linalg
import statsmodels.regression.linear_model as lm
from sklearn import datasets
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LassoCV, LassoLarsCV, LassoLarsIC, RandomizedLasso, lasso_stability_path, ElasticNet
from sklearn.feature_selection import f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import auc, precision_recall_curve, mean_squared_error
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.utils.extmath import pinvh
from sklearn.exceptions import ConvergenceWarning
import warnings
from zipfile import ZipFile
from datetime import datetime
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

'''
def mutual_incoherence(X_relevant, X_irelevant):
	"""Mutual incoherence, as defined by formula (26a) of [Wainwright2006].
	"""
	projector = np.dot(np.dot(X_irelevant.T, X_relevant), pinvh(np.dot(X_relevant.T, X_relevant)))
	return np.max(np.abs(projector).sum(axis=1))



def plot_ic_criterion(model, name, color):
	alpha_ = model.alpha_
	alphas_ = model.alphas_
	criterion_ = model.criterion_
	plt.plot(-np.log10(alphas_), criterion_, '--', color=color,
	         linewidth=3, label='%s criterion' % name)
	plt.axvline(-np.log10(alpha_), color=color, linewidth=3,
	            label='alpha: %s estimate' % name)
	plt.xlabel('-log(alpha)')
	plt.ylabel('criterion')


def filter_data(background):
	nRow = len(background) 
	nCol = len(background.iloc[0,:])

	X_all = background.drop(['challengeID', 'idnum'], axis=1)
	X_train = X_all.iloc[:2121,:]

	Y_train = pd.read_csv("train.csv", low_memory=False)
	all_grit = Y_train['grit'] #This is a Series

	#Remove rows where grit is NA
	grit_defined = np.where(all_grit.notnull())
	grit = all_grit.iloc[grit_defined]
	X_train_grit = X_train.iloc[grit_defined]

	return X_train_grit, grit

def feature_selection(X, y):
	#alphas = np.logspace(-3,-1,21)
	print "Feature Selection"
	X = X.as_matrix()
	y = y.as_matrix()
	with warnings.catch_warnings():
		warnings.simplefilter('ignore', UserWarning)
		warnings.simplefilter('ignore', ConvergenceWarning)
		print "\tLasso Stability Path"
		#alpha_grid, scores_path = lasso_stability_path(X, y, random_state=43, eps=0.05)
		print "\tRandomizedLasso"
		lasso = RandomizedLasso(alpha='aic', random_state=39)
		lasso.fit(X,y)		

	# plt.figure()
	# # We plot the path as a function of alpha/alpha_max to the power 1/3: the
	# # power 1/3 scales the path less brutally than the log, and enables to
	# # see the progression along the path
	# #hg = plt.plot(alpha_grid[1:] ** .333, scores_path[coef != 0].T[1:], 'r')
	# hb = plt.plot(alpha_grid[1:] ** .333, scores_path.T[1:], 'k')
	# ymin, ymax = plt.ylim()
	# plt.xlabel(r'$(\alpha / \alpha_{max})^{1/3}$')
	# plt.ylabel('Stability score: proportion of times selected')
	# plt.title('Stability Scores Path')# - Mutual incoherence: %.1f' % mi)
	# plt.axis('tight')
	# #plt.legend((hg[0], hb[0]), ('relevant features', 'irrelevant features'),
	# #           loc='best')
	# plt.show()

	return lasso

def gen_submission(pred):

	pred.to_csv("prediction.csv", index=False)
	with ZipFile( str('Submission' + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '.zip'), 'w') as myzip:
		myzip.write('prediction.csv')
		myzip.write('narrative.txt')
		myzip.write('ffc.py')

def main():
	#Impute data.
	#fillMissing('background.csv', 'output.csv') #Comment this out after one run

	background = pd.read_csv("output.csv", low_memory=False)
	prediction = pd.read_csv("prediction_old.csv", low_memory=False)

	X,y = filter_data(background)

	randomized_lasso = feature_selection(X,y)
	Xf = X.loc[:,randomized_lasso.get_support()]

	#Regular OLS on grit
	with warnings.catch_warnings():
		warnings.simplefilter('ignore', UserWarning)
		warnings.simplefilter('ignore', ConvergenceWarning)
		olsf = lm.OLS(y, Xf)
		print "OLS"
		print olsf.fit().summary()

		'''Xstd = X.sub(X.mean(1), axis=0).div(X.std(1), axis=0)

		#LASSO on grit
		ols = lm.OLS(y,Xstd)
		print "LASSO"
		fit_lasso = ols.fit_regularized()
		print fit_lasso.summary()

		print "ElasticNet"
		fit_elastic = ols.fit_regularized(alpha=1.0, L1_wt=0.5)
		print fit_lasso.summary()
		'''

	#Prediction
	testf = randomized_lasso.transform(background.drop(['challengeID','idnum'],axis=1))
	grit_predict = olsf.fit().predict(testf)
	grit_predict_round = np.round(grit_predict*4)/4 #Round to nearest 0.25
	grit_predict_round = np.where(grit_predict_round < 4.0, grit_predict_round, 4.0) #Bounds
	grit_predict_round = np.where(grit_predict_round > 1.0, grit_predict_round, 1.0) #Bounds

	prediction['grit'] = grit_predict_round

	print "Training MSE (w/rounding): " + str(mean_squared_error(y.as_matrix(), prediction.ix[y.index, 'grit'].as_matrix()))

	gen_submission(prediction)

	#plt.draw()
	#plt.pause(0.001)

def fillMissing(inputcsv, outputcsv):    
    # read input csv - takes time
    df = pd.read_csv(inputcsv, low_memory=False)
    # Fix date bug
    df.cf4fint = ((pd.to_datetime(df.cf4fint) - pd.to_datetime('1960-01-01')) / np.timedelta64(1, 'D')).astype(int)

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
    dfn.to_csv(outputcsv, index=False)
    
if __name__ == "__main__":
	main()


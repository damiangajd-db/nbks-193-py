# Databricks notebook source
# MAGIC %md
# MAGIC ## Trying out a linear model: 
# MAGIC 
# MAGIC Author: Alexandru Papiu ([Blog](https://apapiu.github.io/), [GitHub](https://github.com/apapiu))
# MAGIC  
# MAGIC If you use parts of this notebook in your own scripts, please give some sort of credit (for example link back to this). Thanks!
# MAGIC 
# MAGIC 
# MAGIC There have been a few [great](https://www.kaggle.com/comartel/house-prices-advanced-regression-techniques/house-price-xgboost-starter/run/348739)  [scripts](https://www.kaggle.com/zoupet/house-prices-advanced-regression-techniques/xgboost-10-kfolds-with-scikit-learn/run/357561) on [xgboost](https://www.kaggle.com/tadepalli/house-prices-advanced-regression-techniques/xgboost-with-n-trees-autostop-0-12638/run/353049) already so I'd figured I'd try something simpler: a regularized linear regression model. Surprisingly it does really well with very little feature engineering. The key point is to to log_transform the numeric variables since most of them are skewed.

# COMMAND ----------

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib

import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.stats.stats import pearsonr


%config InlineBackend.figure_format = 'retina' #set 'png' here when working on notebook
%matplotlib inline

# COMMAND ----------

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

# COMMAND ----------

train.head()

# COMMAND ----------

all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],
                      test.loc[:,'MSSubClass':'SaleCondition']))

# COMMAND ----------

# MAGIC %md
# MAGIC ###Data preprocessing: 
# MAGIC We're not going to do anything fancy here: 
# MAGIC  
# MAGIC - First I'll transform the skewed numeric features by taking log(feature + 1) - this will make the features more normal    
# MAGIC - Create Dummy variables for the categorical features    
# MAGIC - Replace the numeric missing values (NaN's) with the mean of their respective columns

# COMMAND ----------

matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)
prices = pd.DataFrame({"price":train["SalePrice"], "log(price + 1)":np.log1p(train["SalePrice"])})
prices.hist()

# COMMAND ----------

#log transform the target:
train["SalePrice"] = np.log1p(train["SalePrice"])

#log transform skewed numeric features:
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index

all_data[skewed_feats] = np.log1p(all_data[skewed_feats])

# COMMAND ----------

all_data = pd.get_dummies(all_data)

# COMMAND ----------

#filling NA's with the mean of the column:
all_data = all_data.fillna(all_data.mean())

# COMMAND ----------

# MAGIC %md
# MAGIC Note that above I use all the data to compute the mean value that is then used for imputation. This is a flawed approach since it introduces some minor data leakage. However I decided to leave it here like this to showcase some common mistakes we all make :)

# COMMAND ----------

#creating matrices for sklearn:
X_train = all_data[:train.shape[0]]
X_test = all_data[train.shape[0]:]
y = train.SalePrice

# COMMAND ----------

# MAGIC %md
# MAGIC ### Models
# MAGIC 
# MAGIC Now we are going to use regularized linear regression models from the scikit learn module. I'm going to try both l_1(Lasso) and l_2(Ridge) regularization. I'll also define a function that returns the cross-validation rmse error so we can evaluate our models and pick the best tuning par

# COMMAND ----------

from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score

def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, X_train, y, scoring="neg_mean_squared_error", cv = 5))
    return(rmse)

# COMMAND ----------

model_ridge = Ridge()

# COMMAND ----------

# MAGIC %md
# MAGIC The main tuning parameter for the Ridge model is alpha - a regularization parameter that measures how flexible our model is. The higher the regularization the less prone our model will be to overfit. However it will also lose flexibility and might not capture all of the signal in the data.

# COMMAND ----------

alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]
cv_ridge = [rmse_cv(Ridge(alpha = alpha)).mean() 
            for alpha in alphas]

# COMMAND ----------

cv_ridge = pd.Series(cv_ridge, index = alphas)
cv_ridge.plot(title = "Validation - Just Do It")
plt.xlabel("alpha")
plt.ylabel("rmse")

# COMMAND ----------

# MAGIC %md
# MAGIC Note the U-ish shaped curve above. When alpha is too large the regularization is too strong and the model cannot capture all the complexities in the data. If however we let the model be too flexible (alpha small) the model begins to overfit. A value of alpha = 10 is about right based on the plot above.

# COMMAND ----------

cv_ridge.min()

# COMMAND ----------

# MAGIC %md
# MAGIC So for the Ridge regression we get a rmsle of about 0.127
# MAGIC 
# MAGIC Let' try out the Lasso model. We will do a slightly different approach here and use the built in Lasso CV to figure out the best alpha for us. For some reason the alphas in Lasso CV are really the inverse or the alphas in Ridge.

# COMMAND ----------

model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(X_train, y)

# COMMAND ----------

rmse_cv(model_lasso).mean()

# COMMAND ----------

# MAGIC %md
# MAGIC Nice! The lasso performs even better so we'll just use this one to predict on the test set. Another neat thing about the Lasso is that it does feature selection for you - setting coefficients of features it deems unimportant to zero. Let's take a look at the coefficients:

# COMMAND ----------

coef = pd.Series(model_lasso.coef_, index = X_train.columns)

# COMMAND ----------

print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")

# COMMAND ----------

# MAGIC %md
# MAGIC Good job Lasso.  One thing to note here however is that the features selected are not necessarily the "correct" ones - especially since there are a lot of collinear features in this dataset. One idea to try here is run Lasso a few times on boostrapped samples and see how stable the feature selection is.

# COMMAND ----------

# MAGIC %md
# MAGIC We can also take a look directly at what the most important coefficients are:

# COMMAND ----------

imp_coef = pd.concat([coef.sort_values().head(10),
                     coef.sort_values().tail(10)])

# COMMAND ----------

matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind = "barh")
plt.title("Coefficients in the Lasso Model")

# COMMAND ----------

# MAGIC %md
# MAGIC The most important positive feature is `GrLivArea` -  the above ground area by area square feet. This definitely make sense. Then a few other  location and quality features contributed positively. Some of the negative features make less sense and would be worth looking into more - it seems like they might come from unbalanced categorical variables.
# MAGIC 
# MAGIC Also note that unlike the feature importance you'd get from a random forest these are _actual_ coefficients in your model - so you can say precisely why the predicted price is what it is. The only issue here is that we log_transformed both the target and the numeric features so the actual magnitudes are a bit hard to interpret and also the relationship is now multiplicative not additive.

# COMMAND ----------

#let's look at the residuals as well:
matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)

preds = pd.DataFrame({"preds":model_lasso.predict(X_train), "true":y})
preds["residuals"] = preds["true"] - preds["preds"]
preds.plot(x = "preds", y = "residuals",kind = "scatter")

# COMMAND ----------

# MAGIC %md
# MAGIC The residual plot looks pretty good.To wrap it up let's predict on the test set and submit on the leaderboard:

# COMMAND ----------

# MAGIC %md
# MAGIC ### Adding an xgboost model:

# COMMAND ----------

# MAGIC %md
# MAGIC Let's add an xgboost model to our linear model to see if we can improve our score:

# COMMAND ----------

import xgboost as xgb

# COMMAND ----------


dtrain = xgb.DMatrix(X_train, label = y)
dtest = xgb.DMatrix(X_test)

params = {"max_depth":2, "eta":0.1}
model = xgb.cv(params, dtrain,  num_boost_round=500, early_stopping_rounds=100)

# COMMAND ----------

model.loc[30:,["test-rmse-mean", "train-rmse-mean"]].plot()

# COMMAND ----------

model_xgb = xgb.XGBRegressor(n_estimators=360, max_depth=2, learning_rate=0.1) #the params were tuned using xgb.cv
model_xgb.fit(X_train, y)

# COMMAND ----------

xgb_preds = np.expm1(model_xgb.predict(X_test))
lasso_preds = np.expm1(model_lasso.predict(X_test))

# COMMAND ----------

predictions = pd.DataFrame({"xgb":xgb_preds, "lasso":lasso_preds})
predictions.plot(x = "xgb", y = "lasso", kind = "scatter")

# COMMAND ----------

# MAGIC %md
# MAGIC Many times it makes sense to take a weighted average of uncorrelated results - this usually imporoves the score although in this case it doesn't help that much. Here we will just do a weighted average based on some randomly picked value. We could also do a CV grid search here as well.

# COMMAND ----------

preds = 0.7*lasso_preds + 0.3*xgb_preds

# COMMAND ----------

solution = pd.DataFrame({"id":test.Id, "SalePrice":preds})
solution.to_csv("ridge_sol.csv", index = False)

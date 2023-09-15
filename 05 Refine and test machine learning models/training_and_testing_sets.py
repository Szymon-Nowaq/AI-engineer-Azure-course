import pandas as pd
import numpy as np
import graphing
import statsmodels.formula.api as smf
import sklearn.model_selection as sms
import sklearn.metrics as skm
from matplotlib import pyplot as plt

req_cols = ['rescues_last_year', 'weight_last_year']
data = pd.read_csv('05 Refine and test machine learning models/dog-training.csv', delimiter='\t', usecols=req_cols)

#checking if there is relation between variables
model = smf.ols(formula='rescues_last_year ~ weight_last_year', data=data).fit()
graphing.scatter_2D(data, label_x='weight_last_year', label_y='rescues_last_year', trendline = lambda x: model.params[1]*x + model.params[0], show=False)

#spilitng data to avoid overfitting model. random state - it must be specified if we want to split data in the same way at every program launch
train, test = sms.train_test_split(data, train_size=0.7, random_state=21)
df1 = pd.DataFrame(data=train)
df1['color'] = 'red'
df2 = pd.DataFrame(data=test)
df2['color'] = 'blue'
df = pd.concat([df1,df2])
plt.scatter(df.weight_last_year, df.rescues_last_year, c = df.color)
#plt.show()

#tak trochę szybciej
plot_set = pd.concat([train,test])
plot_set["Dataset"] = ["train"] * len(train) + ["test"] * len(test)
graphing.scatter_2D(plot_set, "weight_last_year", "rescues_last_year", "Dataset", trendline = lambda x: model.params[1] * x + model.params[0], show=False)

# Training set
model_train = smf.ols(formula='rescues_last_year ~ weight_last_year', data=train).fit()
graphing.scatter_2D(train, "weight_last_year", "rescues_last_year", trendline = lambda x: model_train.params[1] * x + model_train.params[0], show=False)

#calculating mean squared error
MSE_train = skm.mean_squared_error(train.rescues_last_year, model_train.predict(train.weight_last_year))
print('Mean squared error (train): ', MSE_train)

#test set and calculating MSE, we still use model from previous set
graphing.scatter_2D(test, "weight_last_year", "rescues_last_year", trendline = lambda x: model_train.params[1] * x + model_train.params[0], show=False)
MSE_test = skm.mean_squared_error(test.rescues_last_year, model_train.predict(test.weight_last_year))
print('Mean squared error (test): ', MSE_test)

#testing where MSE is greater, train or test | lp - loop
MSE_lp = pd.DataFrame({'Test':[], 'Train':[]})
ite_num = 0
for random in range(ite_num):
    train_lp, test_lp = sms.train_test_split(data, train_size=0.7, random_state=random)
    model_lp = smf.ols(formula='rescues_last_year ~ weight_last_year', data=train_lp).fit()
    MSE_lp.loc[random] = [skm.mean_squared_error(test_lp.rescues_last_year, model_lp.predict(test_lp.weight_last_year)), skm.mean_squared_error(train_lp.rescues_last_year, model_lp.predict(train_lp.weight_last_year))]
MSE_train_avg = MSE_lp['Train'].mean()
MSE_test_avg = MSE_lp['Test'].mean()
#print('avg MSE test: ', MSE_test_avg)
#print('avg MSE train: ', MSE_train_avg)

#testing our model with new dataset
swiss_doggos = pd.read_csv('05 Refine and test machine learning models/dog-training-swiss.csv', delimiter='\t', usecols=req_cols)
graphing.scatter_2D(swiss_doggos, "weight_last_year", "rescues_last_year", trendline = lambda x: model.params[1] * x + model.params[0], show=True)
MSE_new = skm.mean_squared_error(swiss_doggos['rescues_last_year'], model_train.predict(swiss_doggos['weight_last_year']))
print('Mean squared error (new dataset): ', MSE_new)
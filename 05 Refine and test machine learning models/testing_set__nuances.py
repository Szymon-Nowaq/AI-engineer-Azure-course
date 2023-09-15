import pandas as pd
import numpy as np
import graphing
import statsmodels.formula.api as smf
import sklearn.model_selection as sms
import sklearn.metrics as skm
from matplotlib import pyplot as plt

req_cols = ['rescues_last_year', 'weight_last_year']
doggos = pd.read_csv('05 Refine and test machine learning models/dog-training.csv', delimiter='\t', usecols=req_cols)

# sprawdzanie jakie proporcje dla treningu i testów mogą się dobrze sprawdzić
# podział na kilka opcji i wyświetlenie ich dystrybucji pozwala wykluczyć te opcje
# w których brakuje lub jest wyraźnie mniejsze zagęszczenie danych w jakimś zakresie  
train_5050, test_5050 = sms.train_test_split(doggos, test_size=0.5, random_state=2)
train_6040, test_6040 = sms.train_test_split(doggos, test_size=0.4, random_state=2)
train_7030, test_7030 = sms.train_test_split(doggos, test_size=0.3, random_state=2)
train_8020, test_8020 = sms.train_test_split(doggos, test_size=0.2, random_state=2)

train_5050, test_5050 = train_5050.assign(Set='train'), test_5050.assign(Set='test')
train_6040, test_6040 = train_6040.assign(Set='train'), test_6040.assign(Set='test')
train_7030, test_7030 = train_7030.assign(Set='train'), test_7030.assign(Set='test')
train_8020, test_8020 = train_8020.assign(Set='train'), test_8020.assign(Set='test')

df_5050 = pd.concat([train_5050, test_5050])
df_6040 = pd.concat([train_6040, test_6040])
df_7030 = pd.concat([train_7030, test_7030])
df_8020 = pd.concat([train_8020, test_8020])

graphing.scatter_2D(df_5050, "weight_last_year", "rescues_last_year", title="50:50 split", label_colour="Set", show=False)
graphing.scatter_2D(df_6040, "weight_last_year", "rescues_last_year", title="60:40 split", label_colour="Set", show=False)
graphing.scatter_2D(df_7030, "weight_last_year", "rescues_last_year", title="70:30 split", label_colour="Set", show=False)
graphing.scatter_2D(df_8020, "weight_last_year", "rescues_last_year", title="80:20 split", label_colour="Set", show=False)

train_5050 = train_5050.assign(Split='50:50')
train_6040 = train_6040.assign(Split='60:40')
train_7030 = train_7030.assign(Split='70:30')
train_8020 = train_8020.assign(Split='80:20')

split_df = pd.concat([train_5050,train_6040,train_7030,train_8020], axis=0)

graphing.multiple_histogram(split_df, label_x="rescues_last_year", label_group="Split", show=False)

# how models are fit with diffrent splits
def train_and_test_model(name, train, test):
    #train
    model = smf.ols(formula='rescues_last_year ~ weight_last_year', data=train).fit()
    #test
    mse = skm.mean_squared_error(test['rescues_last_year'], model.predict(test['weight_last_year']))
    #print(name, 'mean squader error: ', mse)
    return model

model_5050 = train_and_test_model('50/50', train_5050, test_5050)
model_6040 = train_and_test_model('60/40', train_6040, test_6040)
model_7030 = train_and_test_model('70/30', train_7030, test_7030)
model_8020 = train_and_test_model('80/20', train_8020, test_8020)


swiss_doggos = pd.read_csv('05 Refine and test machine learning models/dog-training-swiss.csv', delimiter='\t', usecols=req_cols)
graphing.scatter_2D(swiss_doggos, 'rescues_last_year', 'weight_last_year', show=False)

mse_swiss_5050 = skm.mean_squared_error(swiss_doggos['rescues_last_year'], model_5050.predict(swiss_doggos['weight_last_year']))
mse_swiss_6040 = skm.mean_squared_error(swiss_doggos['rescues_last_year'], model_6040.predict(swiss_doggos['weight_last_year']))
mse_swiss_7030 = skm.mean_squared_error(swiss_doggos['rescues_last_year'], model_7030.predict(swiss_doggos['weight_last_year']))
mse_swiss_8020 = skm.mean_squared_error(swiss_doggos['rescues_last_year'], model_8020.predict(swiss_doggos['weight_last_year']))
print('mean squared error (swiss) 50:50', mse_swiss_5050)
print('mean squared error (swiss) 60:40', mse_swiss_6040)
print('mean squared error (swiss) 70:30', mse_swiss_7030)
print('mean squared error (swiss) 80:20', mse_swiss_8020)
import numpy as np
import pandas as pd
import graphing
from m1b_gradient_descent import gradient_descent
from matplotlib import pyplot as plt
import plotly.express as px

aval_dogs = pd.read_csv('05 Refine and test machine learning models/dog-training.csv', delimiter='\t')

#training model
model = gradient_descent(aval_dogs.month_old_when_trained, aval_dogs.mean_rescues_per_year, learning_rate=5E-4, number_of_iterations=8000)
graphing.scatter_2D(aval_dogs, "month_old_when_trained", "mean_rescues_per_year", trendline=model.predict, show=True)

#normalize model before train 
aval_dogs['std_age_when_trained'] = (aval_dogs.month_old_when_trained - aval_dogs.month_old_when_trained.mean()) / np.std(aval_dogs.month_old_when_trained)

#fig = px.box(aval_dogs,y=["month_old_when_trained", "standardized_age_when_trained"])
#fig.write_html('fig.html', auto_open=True)
model_norm = gradient_descent(aval_dogs.std_age_when_trained, aval_dogs.mean_rescues_per_year, learning_rate=5E-4, number_of_iterations=8000)
graphing.scatter_2D(aval_dogs, 'std_age_when_trained', "mean_rescues_per_year", trendline=model_norm.predict, show=True)

#comparing costs of training normalized and not model
costNone = model.cost_history
costNorm = model_norm.cost_history

df1 = pd.DataFrame({'cost': costNone, 'Model':'None'})
df2 = pd.DataFrame({'cost': costNorm, 'Model':'Normalized'})
df1['number_of_ite'] = df1.index + 1
df2['number_of_ite'] = df2.index + 1
df = pd.concat([df1,df2])
print(df.head())
graphing.scatter_2D(df, label_x='number_of_ite', label_y='cost',show=True)
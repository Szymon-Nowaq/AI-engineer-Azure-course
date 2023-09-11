import pandas as pd
from matplotlib import pyplot as plt
import statsmodels.formula.api as smf
import numpy as np

doggy_df = pd.read_csv('doggy.csv', delimiter = '\t', header='infer')

# Multiple linear reggresion
model = smf.ols(formula = 'core_temperature ~ age + male', data = doggy_df).fit()
#print("R-squared:", model.rsquared)
#print("params:", model.params)

x = doggy_df.age
y = doggy_df.male
z = doggy_df.core_temperature

#meshgridowaie - wyznaczanie siatki - wszystkich możliwych kombinacji 
X, Y = np.meshgrid(np.linspace(min(x), max(x)), np.array([0,1]))

#powierchnia Z - tablica 2D
Z = model.predict(exog=pd.DataFrame({'age': X.flatten(), 'male': Y.flatten()})).values.reshape(X.shape) 

# Tworzenie wykresu 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# Etykiety osi
ax.set_xlabel('X - age')
ax.set_ylabel('Y - male')
ax.set_zlabel('Z - temp')

# całego wykresu
ax.plot_surface(X, Y, Z, alpha=0.5, cmap='coolwarm', label='Surface')

#plt.show()

#dane na temat modelu
#print(model.summary())
import pandas as pd
from matplotlib import pyplot as plt
import statsmodels.formula.api as smf
import numpy as np



doggy_df = pd.read_csv('doggy.csv', delimiter = '\t', header='infer')


# reminder - linear regression, in this case regression line should be curved
#fig, ax = plt.subplots(1,2)
linear_model = smf.ols(formula= 'core_temperature ~ protein_content_of_last_meal', data= doggy_df).fit()
# ax[0].scatter(doggy_df['protein_content_of_last_meal'], doggy_df['core_temperature'], c='blue')
# ax[0].plot(doggy_df['protein_content_of_last_meal'], linear_model.params[1] * doggy_df['protein_content_of_last_meal'] + linear_model.params[0], color='red')

print("R-squared:", linear_model.rsquared)


# generating polynominal regression line
polynominal_formula = "core_temperature ~ protein_content_of_last_meal + I(protein_content_of_last_meal**2)"
polynomial_model = smf.ols(formula = polynominal_formula, data=doggy_df).fit()

#sortowanie dla wykresu - ważne!
x = np.sort(doggy_df['protein_content_of_last_meal'])

# ax[1].scatter(doggy_df['protein_content_of_last_meal'], doggy_df['core_temperature'], c='blue')
# ax[1].plot(x, polynomial_model.params[2] * x ** 2 + polynomial_model.params[1] * x + polynomial_model.params[0], color='red')
plt.show()

print("R-squared:", polynomial_model.rsquared)


# 3D
fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
# ax.set_xlabel('x1')
# ax.set_ylabel('x2')
# ax.set_zlabel('temp')

x1_values = np.array([min(doggy_df.protein_content_of_last_meal), max(doggy_df.protein_content_of_last_meal)]),
x2_values = np.array([min(doggy_df.protein_content_of_last_meal)**2, max(doggy_df.protein_content_of_last_meal)**2]),

X1, X2 = np.meshgrid(x1_values, x2_values)
Z = polynomial_model.params[0] + (polynomial_model.params[1] * X1) + (polynomial_model.params[2] * X2)

#ax.plot_surface(X1, X2, Z, alpha=0.5, cmap='coolwarm', label='Surface')
#plt.show()


# extrapolating
fig, ax = plt.subplots(1,2)
linear_model = smf.ols(formula= 'core_temperature ~ protein_content_of_last_meal', data= doggy_df).fit()
ax[0].scatter(doggy_df['protein_content_of_last_meal'], doggy_df['core_temperature'], c='blue')
ax[0].plot(doggy_df['protein_content_of_last_meal'], linear_model.params[1] * doggy_df['protein_content_of_last_meal'] + linear_model.params[0], color='red')
ax[0].set_xlim([0,100])

print("R-squared:", linear_model.rsquared)


# generating polynominal regression line
polynominal_formula = "core_temperature ~ protein_content_of_last_meal + I(protein_content_of_last_meal**2)"
polynomial_model = smf.ols(formula = polynominal_formula, data=doggy_df).fit()

#sortowanie dla wykresu - ważne!
x = np.sort(doggy_df['protein_content_of_last_meal'])

ax[1].scatter(doggy_df['protein_content_of_last_meal'], doggy_df['core_temperature'], c='blue')
ax[1].plot(x, polynomial_model.params[2] * x ** 2 + polynomial_model.params[1] * x + polynomial_model.params[0], color='red')
ax[1].set_xlim([0,100])

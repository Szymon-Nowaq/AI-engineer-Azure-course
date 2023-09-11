import pandas as pd
from matplotlib import pyplot as plt
import statsmodels.formula.api as smf
import numpy as np

doggy_df = pd.read_csv('doggy.csv', delimiter = '\t', header='infer')
 
#print(doggy_df)
#plt.hist(doggy_df.age, bins=10)
#plt.show()
#plt.hist(doggy_df.core_temperature, bins=10)
#plt.show()

#doggy_df.plot(x='age', y='core_temperature', kind='scatter', figsize=(10,8))
#plt.show()

#regresja liniowa, formula - specjalny zapis mówiący że rozważamy funkcję CT(age)
formula = "core_temperature ~ age"
model = smf.ols(formula=formula, data=doggy_df).fit()
#plt.scatter(doggy_df['age'], doggy_df['core_temperature'])
#plt.plot(doggy_df['age'], model.params[1] * doggy_df['age'] + model.params[0], color='red')
#plt.show()

def estimate_temp(age):
    print("estimated temperature for doggo aged ", age, ' is: ', round(model.params[1] * age + model.params[0],2))

for i in range(1,10):
    estimate_temp(i)

#predicting core_temp by other features
# fig, axis = plt.subplots(3,1,figsize=(10,12))

# axis[0].scatter( doggy_df["body_fat_percentage"], doggy_df["core_temperature"])
# axis[0].set_title('body_fat_percentage')
# axis[1].scatter( doggy_df["protein_content_of_last_meal"], doggy_df["core_temperature"])
# axis[1].set_title('protein_content_of_last_meal')
# axis[2].scatter( doggy_df["age"], doggy_df["core_temperature"])
# axis[2].set_title('age')

# plt.show()

for feature in ["male", "age", "protein_content_of_last_meal", "body_fat_percentage"]:
    # Perform linear regression. This method takes care of the entire fitting procedure for us. 
    simple_model = smf.ols(formula = "core_temperature ~ " + feature, data = doggy_df).fit()

    print(feature)
    print("R-squared:", simple_model.rsquared)
    
    # Show a graph of the result
    plt.scatter(doggy_df[feature], doggy_df["core_temperature"])
    plt.plot(doggy_df[feature], simple_model.params[1] * doggy_df[feature] + simple_model.params[0], color='red')
    plt.show()

# w jaki sposób obliczny jest r^2? zobrazowanie modelu naiwnego i wytrenowanego
naive_model = smf.ols(formula = 'core_temperature ~ age', data = doggy_df).fit()
trained_model = smf.ols(formula = 'core_temperature ~ age', data = doggy_df).fit()
naive_model.params[0] = doggy_df['core_temperature'].mean()
naive_model.params[1] = 0

print("naive r^2: ", naive_model.rsquared)
print("trained r^2: ", trained_model.rsquared)

plt.scatter(doggy_df['age'], doggy_df["core_temperature"])
plt.plot(doggy_df['age'], naive_model.params[1] * doggy_df['age'] + naive_model.params[0], color='red')
plt.plot(doggy_df['age'], trained_model.params[1] * doggy_df['age'] + trained_model.params[0], color='green')
plt.show()


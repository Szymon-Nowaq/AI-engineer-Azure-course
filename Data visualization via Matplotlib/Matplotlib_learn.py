import pandas as pd 
from matplotlib import pyplot as plt

#przygotowanie csv tak jak to było w poprzednim temacie
df_students = pd.read_csv('.\Data exploration via NumPy and Pandas\grades.csv', delimiter=',',header='infer')
df_students = df_students.dropna(axis = 0, how = 'any')
passes = pd.Series(df_students['Grade'] >= 60)
df_students = pd.concat([df_students, passes.rename("Pass")], axis=1)
#print(df_students)

#generowanie wykresu, bar - słupkowy:  
#plt.bar(x=df_students.Name, height=df_students.Grade, color='blue')

#scatter - punktowy
#plt.scatter(x=df_students.Name, y=df_students.Grade, color='blue')

#plot - liniowy
#plt.plot(df_students.Name, df_students.Grade)

#hist - histogram, pokazuje jak często osiągane byy dane wyniki
#plt.hist(df_students.Grade, bins=22)

#boxplot - 
#plt.boxplot(df_students['Grade'])

#zabawa kolorkami itd
#plt.title('Student Grades')
##plt.xlabel('Student')
#plt.ylabel('Grade')
#plt.grid(color='#95a5a6', linestyle='--', linewidth=2, axis='y', alpha=0.7)
#plt.xticks(rotation=60)

#plt.show()

#figures - before it was created impicity, explicity here: 
fig = plt.figure(figsize=(10,8)) #determine the size of window
#plt.bar(x=df_students.Name, height=df_students.Grade, color='blue')

#plt.title('Student Grades')
#plt.xlabel('Student')
#plt.ylabel('Grade')
#plt.grid(color='#95a5a6', linestyle='--', linewidth=2, axis='y', alpha=0.7)
#plt.xticks(rotation=60)

#plt.show()

#fig - przechowuje całość, axis - tablica wykresów
#fig, axis = plt.subplots(1, 2, figsize=(8,3)) #axis - 1 wiersz, dwie kolumny

#tworzenie pierwszego wykresu
#axis[0].bar(x=df_students.Name, height = df_students.Grade)
#axis[0].set_title('Grades')
#axis[0].set_xticklabels(df_students.Name, rotation=90)

#drugi wykres, pie - wykres kołowy
#pass_counts = df_students['Pass'].value_counts() #pass_counts przechowuje to ile było T i F w kolumnie Pass
#axis[1].pie(pass_counts, labels=pass_counts)
#axis[1].set_title('Passing Grade')
#axis[1].legend(pass_counts.keys().tolist())

fig.suptitle("Student data") #nazwa całości

#plt.show()

#histogram jeszcze raz, obrazuje dystrybucje danych
var_data = df_students['Grade']
#plt.hist(var_data)
plt.title('Data Distribution')
plt.xlabel('Value')
plt.ylabel('Frequency')

#plt.show()

#rzeczy z gimby 
min = var_data.min()
max = var_data.max()
mean = var_data.mean()
med = var_data.median()
mod = var_data.mode()[0]
#print('Minimum:{:.2f}\nMean:{:.2f}\nMedian:{:.2f}\nMode:{:.2f}\nMaximum:{:.2f}\n'.format(min,mean,med,mod,max))

#rzeczy z gimby na wykresie
plt.axvline(x=min, color = 'gray', linestyle='dashed', linewidth = 2)
plt.axvline(x=mean, color = 'cyan', linestyle='dashed', linewidth = 2)
plt.axvline(x=med, color = 'red', linestyle='dashed', linewidth = 2)
plt.axvline(x=mod, color = 'yellow', linestyle='dashed', linewidth = 2)
plt.axvline(x=max, color = 'gray', linestyle='dashed', linewidth = 2)

#plt.show()

#function showing both histogram and boxplot
def show_distribution(data):
    fig, axis = plt.subplots(2, 1, figsize=(10,8)) #dwa wiersze, jedna koumna

    min = var_data.min()
    max = var_data.max()
    mean = var_data.mean()
    med = var_data.median()
    mod = var_data.mode()[0]
    
    axis[0].hist(data) #histogram
    axis[0].set_ylabel("Frequency")
    #gimbalinie:
    axis[0].axvline(x=min, color = 'gray', linestyle='dashed', linewidth = 2)
    axis[0].axvline(x=mean, color = 'cyan', linestyle='dashed', linewidth = 2)
    axis[0].axvline(x=med, color = 'red', linestyle='dashed', linewidth = 2)
    axis[0].axvline(x=mod, color = 'yellow', linestyle='dashed', linewidth = 2)
    axis[0].axvline(x=max, color = 'gray', linestyle='dashed', linewidth = 2)
    #boxplot:
    axis[1].boxplot(data, vert=False)
    axis[1].set_xlabel("Value")

    fig.suptitle('Data Distribution')
    plt.show()

#show_distribution(var_data)

#denity shows on a diagram how the diagram should look like (based on sampled data)if we would examine all population
def show_density(data):
    fig = plt.figure(figsize=(10,4))

    data.plot.density()
    plt.title('Data Density')

    # gimbolinie
    plt.axvline(x=data.mean(), color = 'cyan', linestyle='dashed', linewidth = 2)
    plt.axvline(x=data.median(), color = 'red', linestyle='dashed', linewidth = 2)
    plt.axvline(x=data.mode()[0], color = 'yellow', linestyle='dashed', linewidth = 2)

    plt.show()

col = df_students['Grade']
show_density(col)
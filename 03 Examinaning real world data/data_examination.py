import pandas as pd
from matplotlib import pyplot as plt
import scipy.stats as stats

# stadardowa pocatkowa obróbka csv
df_students = pd.read_csv('grades.csv',delimiter=',',header='infer')
df_students = df_students.dropna(axis=0, how='any')
passes  = pd.Series(df_students['Grade'] >= 60)
df_students = pd.concat([df_students, passes.rename("Pass")], axis=1)

# funkcja z poprzedniego tematu
def show_distribution(var_data):
    # Get statistics
    min_val = var_data.min()
    max_val = var_data.max()
    mean_val = var_data.mean()
    med_val = var_data.median()
    mod_val = var_data.mode()[0]

    print('Minimum:{:.2f}\nMean:{:.2f}\nMedian:{:.2f}\nMode:{:.2f}\nMaximum:{:.2f}\n'.format(min_val,mean_val,med_val,mod_val,max_val))

    # Create a figure for 2 subplots (2 rows, 1 column)
    fig, ax = plt.subplots(2, 1, figsize = (10,4))

    # Plot the histogram   
    ax[0].hist(var_data)
    ax[0].set_ylabel('Frequency')

    # Add lines for the mean, median, and mode
    ax[0].axvline(x=min_val, color = 'gray', linestyle='dashed', linewidth = 2)
    ax[0].axvline(x=mean_val, color = 'cyan', linestyle='dashed', linewidth = 2)
    ax[0].axvline(x=med_val, color = 'red', linestyle='dashed', linewidth = 2)
    ax[0].axvline(x=mod_val, color = 'yellow', linestyle='dashed', linewidth = 2)
    ax[0].axvline(x=max_val, color = 'gray', linestyle='dashed', linewidth = 2)

    # Plot the boxplot   
    ax[1].boxplot(var_data, vert=False)
    ax[1].set_xlabel('Value')

    fig.suptitle('Data Distribution')
    plt.show()

show_distribution(df_students['Grade'])

# dystrybucja StudyHours bardzos się różni od Grade
show_distribution(df_students['StudyHours'])

# jest jeden outliner - wartość 1, usuwamy
col = df_students[df_students.StudyHours > 1]['StudyHours']
show_distribution(col)
# after deleting outliner the plot has normalized

# granice można uzyskać funkcją
# do quantile przekazujemy procent, zwraca liczbę, poniżej której znajduje się podany procent całości danych 
q1 = df_students.StudyHours.quantile(0.1) 
col = df_students[df_students.StudyHours > q1]['StudyHours']
show_distribution(col)

# funkcja z poprzedniego
# pokacuje że dane są right skewed - większość z lewej, kilka max w prawo
def show_density(data):
    fig = plt.figure(figsize=(10,4))

    data.plot.density()
    plt.title('Data Density')

    # gimbolinie
    plt.axvline(x=data.mean(), color = 'cyan', linestyle='dashed', linewidth = 2)
    plt.axvline(x=data.median(), color = 'red', linestyle='dashed', linewidth = 2)
    plt.axvline(x=data.mode()[0], color = 'yellow', linestyle='dashed', linewidth = 2)

    plt.show()

show_density(col)

# inne dane 

for column in ['StudyHours', 'Grade']:
    col = df_students[column]
    range = col.max() - col.min() # rozstaw danych
    var = col.var() # variance - średnia z odległości od śr art podniesionych do kwadratu
    std = col.std() # sqrt(var) - miara "rozrzucenia" danych - standard devation
    print('\n{}:\n - Range: {:.2f}\n - Variance: {:.2f}\n - Std.Dev: {:.2f}'.format(column, range, var, std))


# zobrazowanie jaka część wszystkich wyników będzie sie mieścić w odstępie 1,2,3 std_dev od średniej art
def show_perc_data_in_123_std_dev(col):
    # generowanie funkcji gęstości pradopodobieńtwa
    density = stats.gaussian_kde(col)
    col.plot.density()

    # Get the mean and standard deviation
    std_dev = col.std()
    mean = col.mean()

    # Annotate prostej która zaczyna się i kończy w punktach odległych o 1 std_dev od meanu
    x1 = [mean-std_dev, mean+std_dev] #rozważany przedział
    y1 = density(x1) #
    plt.plot(x1,y1, color='magenta')
    plt.annotate('1 std (68.26%)', (x1[1],y1[1]))

    # to samo dla dwóch odległości
    x2 = [mean-(std_dev*2), mean+(std_dev*2)]
    y2 = density(x2)
    plt.plot(x2,y2, color='green')
    plt.annotate('2 std (95.45%)', (x2[1],y2[1]))

    # i dla trzech
    x3 = [mean-(std_dev*3), mean+(std_dev*3)]
    y3 = density(x3)
    plt.plot(x3,y3, color='orange')
    plt.annotate('3 std (99.73%)', (x3[1],y3[1]))

    # prosta prostopadła do y pokazująca meana
    plt.axvline(col.mean(), color='cyan', linestyle='dashed', linewidth=1)
    # ukrycie osi x i y
    plt.axis('on')
    plt.show()

col = df_students['Grade']
col = df_students['StudyHours']
show_perc_data_in_123_std_dev(col)

# descirbe pokazuje podstawowe staty dostyczące dataframe'u
print(df_students.describe())

# comparing data
df_sample = df_students[df_students.StudyHours > q1]

# comparing numeric (SH) and categorical (Pass) values
# podzielenie danych ze SH na dwie częśc w zależności od T/F Passa, dla obu części boxplot
df_sample.boxplot(column='StudyHours', by='Pass', figsize=(8,5))

# comparing two numeric values
df_sample.plot(x='Name', y=['Grade', 'StudyHours'], kind='bar', figsize=(8,5))
# problem - Grade ma przedział 0-100, SH: 0-16, wykres nie jest proporcjonalny

# skalowanie
from sklearn.preprocessing import MinMaxScaler as mms
scaler = mms()

# kopia dataframe dla scalera, .copy, bo normalne przypisanie idzie przez referencję 
df_for_scaler = df_sample[['Name', 'Grade', 'StudyHours']].copy()
# skalowanie - działa na zasadzie spłaszczenia danych do wartości od 0 do 1 gdzie są to odpowiednio
# min i max z oryginalnej tablicy, reszta dopasowana jest WP
df_for_scaler[['StudyHours', 'Grade']] = scaler.fit_transform(df_for_scaler[['StudyHours', 'Grade']])
df_for_scaler.plot(x='Name', y=['Grade', 'StudyHours'], kind='bar', figsize=(8,5))

# wskaźnik korelacji - w zakresie od -2 do 1, określa jak dane są od siebie zależne 
# jak większe od zera to korelacja pozytywna 
print(df_for_scaler.Grade.corr(df_for_scaler.StudyHours))

# inny sposób na pokazanie korelacji to wykres punktowy
df_sample.plot.scatter(title='Study Time vs Grade', x='StudyHours', y='Grade')

#plt.show()


#laby z fizy ale to python - regresja liniowa 
df_regression = df_sample[['Grade', 'StudyHours']].copy()

#lineregress robi całą robotę, sporo zmiennych zwraca, dajemy mu iksy i igreki
a, b, r, p, std_error = stats.linregress(df_regression['StudyHours'], df_regression['Grade'])
#print('slope: {:.4f}\ny-intercept: {:.4f}'.format(a,b))
#print('f(x) = {:.4f}x + {:.4f}'.format(a,b))

# stwórz nową kolumnę, gdzie będzie f(x)
df_regression['fx'] = (a * df_regression['StudyHours']) + b

# w kolejnej kolumnie różnica między regreasją a pomiarami
df_regression['error'] = abs(df_regression['fx'] - df_regression['Grade'])

#wykresy
df_regression.plot.scatter(x='StudyHours', y='Grade') #punktowy
plt.plot(df_regression['StudyHours'],df_regression['fx'], color='cyan') #liniowy
#plt.show()
#print(df_regression.sort_values('StudyHours'))

def predict_my_grade(study_hours):
    return study_hours*6.3134-17.9164

x = 10
print('studying for ',x,' hours will probably give u grade: ',predict_my_grade(x))
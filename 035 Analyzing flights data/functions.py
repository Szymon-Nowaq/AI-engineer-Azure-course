import pandas as pd 
from matplotlib import pyplot as plt

def show_distribution(data):
    fig, axis = plt.subplots(2, 1, figsize=(10,8)) #dwa wiersze, jedna koumna

    min = data.min()
    max = data.max()
    mean = data.mean()
    med = data.median()
    
    axis[0].hist(data) #histogram
    axis[0].set_ylabel("Frequency")
    #gimbalinie:
    # axis[0].axvline(x=min, color = 'gray', linestyle='dashed', linewidth = 2)
    # axis[0].axvline(x=mean, color = 'cyan', linestyle='dashed', linewidth = 2)
    # axis[0].axvline(x=med, color = 'red', linestyle='dashed', linewidth = 2)
    # axis[0].axvline(x=max, color = 'gray', linestyle='dashed', linewidth = 2)
    #boxplot:
    axis[1].boxplot(data, vert=False)
    axis[1].set_xlabel("Value")

    fig.suptitle('Data Distribution')
    plt.show()

def show_density(data):
    fig = plt.figure(figsize=(10,4))

    data.plot.density()
    plt.title('Data Density')

    # gimbolinie
    # plt.axvline(x=data.mean(), color = 'cyan', linestyle='dashed', linewidth = 2)
    # plt.axvline(x=data.median(), color = 'red', linestyle='dashed', linewidth = 2)
    # plt.axvline(x=data.mode()[0], color = 'yellow', linestyle='dashed', linewidth = 2)

    plt.show()

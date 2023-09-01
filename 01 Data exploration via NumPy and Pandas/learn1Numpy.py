import numpy as np
import pandas as pd

grades = [50,50,47,97,49,3,53,42,26,74,82,62,37,15,70,27,36,35,48,52,63,64]

nP_grades = np.array(grades)

study_hours = [10.0,11.5,9.0,16.0,9.25,1.0,11.5,9.0,8.5,14.5,15.5,
               13.75,9.0,8.0,15.5,8.0,9.0,6.0,10.0,12.0,12.5,12.0]
student_data = np.array([grades, study_hours])
print(grades*3)
print(nP_grades*3)

print(student_data.shape)

avg_grades = student_data[0].mean()
avg_study = student_data[1].mean()

print(avg_grades)
print(avg_study)

df_students = pd.DataFrame({'Name': ['Dan', 'Joann', 'Pedro', 'Rosie', 'Ethan', 'Vicky', 'Frederic', 'Jimmie', 
    'Rhonda', 'Giovanni', 'Francesca', 'Rajab', 'Naiyana', 'Kian', 'Jenny',
    'Jakeem','Helena','Ismat','Anila','Skye','Daniel','Aisha'],
    'StudyHours':student_data[0],
    'Grade':student_data[1]})
print(df_students.loc[df_students['Name']=="Pedro", "Grade"])
print(df_students.loc[0, "Grade"])


#import tablicy z neta, csv - CommaSeperatedVales
df_students = pd.read_csv('grades.csv',delimiter=',',header='infer')
print(df_students.head(10))


# wyszukiwanie pustych miejsc 
print(df_students.isnull()) 
print(df_students.isnull().sum()) #sumuje NaN dla kolumn
print(df_students[df_students.isnull().any(axis=1)]) #wypisanie wierszy z NaN, axis 1 - wiersze, 0 - kolumny

# podmiana NaN-ów przez średnią wartość z kolumny
df_students.StudyHours = df_students.StudyHours.fillna(df_students.StudyHours.mean())

# usuniecie wierszów - axis = 0, w których znajduje się przynajmniej jeden NaN - "how"
df_students = df_students.dropna(axis = 0, how = 'any')

#średnie
mean_study = df_students.StudyHours.mean()
mean_grade = df_students["Grade"].mean()
print(round(mean_study,2))
print(round(mean_grade,2))

#studenci powyżej średniej studyhours
print(df_students[df_students.StudyHours > mean_study])
#ich średnia ocen
print(round(df_students[df_students.StudyHours > mean_study].Grade.mean(),2))

#dodajemy do tablicy info czy zaliczyli
passes = pd.Series(df_students["Grade"] >= 60) #tworzenie serii (listy) na podstawie warunku (zwraca T/F)
df_students = pd.concat([df_students, passes.rename("Pass")], axis = 1) #dodanie serii jako kolumny (axis=1), rename nadaje nazwę

#sortowanie wg wyniku
df_students = df_students.sort_values(by = "Grade", ascending = False)
print(df_students)

#podsumowanie wyników przez grupowanie
print(df_students.groupby(df_students.Pass).Name.count())
print(df_students.groupby(df_students.Pass)[['StudyHours', 'Grade']].mean())
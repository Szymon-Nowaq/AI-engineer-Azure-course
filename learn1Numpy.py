import numpy as np
import pandas as pd

grades = [50,50,47,97,49,3,53,42,26,74,82,62,37,15,70,27,36,35,48,52,63,64]

nP_grades = np.array(grades)

study_hours = [10.0,11.5,9.0,16.0,9.25,1.0,11.5,9.0,8.5,14.5,15.5,
               13.75,9.0,8.0,15.5,8.0,9.0,6.0,10.0,12.0,12.5,12.0]
student_data = np.array([grades, study_hours])
#print(grades*3)
#print(nPgrades*3)

#print(student_data.shape)

avg_grades = student_data[0].mean()
avg_study = student_data[1].mean()

#print(avg_grades)
#print(avg_study)

df_students = pd.DataFrame({'Name': ['Dan', 'Joann', 'Pedro', 'Rosie', 'Ethan', 'Vicky', 'Frederic', 'Jimmie', 
    'Rhonda', 'Giovanni', 'Francesca', 'Rajab', 'Naiyana', 'Kian', 'Jenny',
    'Jakeem','Helena','Ismat','Anila','Skye','Daniel','Aisha'],
    'StudyHours':student_data[0],
    'Grade':student_data[1]})
#print(df_students.loc[df_students['Name']=="Pedro", "Grade"])
#print(df_students.loc[0, "Grade"])

df_students = pd.read_csv('grades.csv',delimiter=',',header='infer')
#print(df_students.head(10))

#print(df_students.isnull())
print(df_students.isnull().sum())
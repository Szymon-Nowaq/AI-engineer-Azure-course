import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from functions import *
from sklearn.preprocessing import MinMaxScaler as mms

#pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)
#req_cols = ['DayOfWeek','Carrier','OriginAirportName','DestAirportID','DestAirportName','DestCity','DepDelay','DepDel15','ArrDelay','ArrDel15']

# zrzucenie tablicy z csv
flights_df = pd.read_csv('flights.csv', delimiter=',', header='infer')

# 1.START BY CLEANING THE DATA.

# 1.a) Identify any null or missing data, and impute appropriate replacement values.
'''# wyszukiwanie i usunięcie wierszów z Nanem
print(flights_df[flights_df.isnull().any(axis=1)][['DepDelay', 'DepDel15']])
flights_df = flights_df.dropna(axis=0, how='any')'''
# lepsza metoda - uzupełnienie NAN-ów wg zasad
#flights_df.at[171, 'DepDelay'] = 20 # kolumnę obok NaN są same zera, sprawdzenie czy poniższy warunek działa
flights_df['DepDel15'] = flights_df.apply(lambda row: 1 if np.isnan(row['DepDel15']) and row['DepDelay'] >= 15 else 0, axis=1)
#print(flights_df[['DepDelay', 'DepDel15']].loc[[171,359,429,545,554]]) #sprawdzenie czy działa (działa :D)

# 1.b) Identify and eliminate any outliers in the DepDelay and ArrDelay columns.
# usunięcie outlinerów z opóźnień - sytuacji gdy przy wylocie jest opóźnienie, a przy przylocie brak

flights_df = flights_df.drop(flights_df[(flights_df.DepDelay > 60) & (flights_df.ArrDelay == 0)].index)
#flights_df.to_csv('flights_clarified.csv')
'''# szukanie outlinerów wykresami
difrence = (abs(flights_df.DepDelay - flights_df.ArrDelay))
#show_distribution(difrence)
skrajne = flights_df[(flights_df.DepDelay > 60) & (flights_df.ArrDelay == 0)][['DepDelay', 'ArrDelay']]
skrajne.plot(y=['DepDelay', 'ArrDelay'], kind='bar', figsize=(10,8))
plt.show() '''


# 2.EXPLORE THE CLEANED DATA

# 2.a) View summary statistics for the numeric fields in the dataset.
flights_df.describe()

# 2.b) Determine the distribution of the DepDelay and ArrDelay columns.
show_distribution(flights_df.DepDelay)
show_distribution(flights_df.ArrDelay)


# 3. USE STATISTICS, AGGREGATE FUNCTIONS, AND VISUALIZATIONS TO ANSWER THE FOLLOWING QUESTIONS:
# 3.a) What are the average (mean) departure and arrival delays?
dep_del_mean = flights_df.DepDelay.mean()
arr_del_mean = flights_df.ArrDelay.mean()
print("Average departure delay: ", round(dep_del_mean,2))
print("Average arrival delay: ", round(arr_del_mean,2))

# 3.b) How do the carriers compare in terms of arrival delay performance?
'''# po mojemu:
airlines = flights_df.Carrier.unique()
dep_means = pd.Series()
arr_means = pd.Series()
for i, carrier in enumerate(airlines):
    dep_means[i] = flights_df[flights_df.Carrier == carrier]['DepDelay'].mean()
for i, carrier in enumerate(airlines):
    arr_means[i] = flights_df[flights_df.Carrier == carrier]['ArrDelay'].mean()
airlines_delays = pd.DataFrame({"Carrier": airlines, "DepDelMean": dep_means, "ArrDelMean": arr_means})'''
# a tu trochę krócej
airlines_delays = pd.DataFrame(data = flights_df.groupby('Carrier')[['DepDelay', 'ArrDelay']].mean().reset_index())
counts_col = pd.Series(flights_df.groupby('Carrier').size().reset_index(drop=True))

#wykresy
airlines_delays = pd.concat([airlines_delays, counts_col.rename("FlightCount")], axis=1)
scaler = mms()
df_for_scaler = airlines_delays.copy()
df_for_scaler[['DepDelay', 'ArrDelay', 'FlightCount']] = scaler.fit_transform(df_for_scaler[['DepDelay', 'ArrDelay', 'FlightCount']])
fig, axes = plt.subplots(2, 1, figsize=(9, 14))
df_for_scaler.plot(x='Carrier', y=['DepDelay', 'ArrDelay', 'FlightCount'], kind='bar', figsize=(9, 7), ax=axes[0])
airlines_delays.plot(x='Carrier', y=['DepDelay', 'ArrDelay'], kind='bar', figsize=(9, 7), ax=axes[1])
#plt.show()

# 3.c) Is there a noticable difference in arrival delays for different days of the week?
days_delays = pd.DataFrame(data = flights_df.groupby('DayOfWeek')['ArrDelay'].mean().reset_index())
days_delays.plot(x='DayOfWeek', y='ArrDelay', kind='bar', figsize = (10,8))
#plt.show()

# 3.d) Which departure airport has the highest average departure delay?
airport_delays = pd.DataFrame(data = flights_df.groupby('OriginAirportName')['DepDelay'].mean().reset_index())
airport_delays.plot(x='OriginAirportName', y='DepDelay', kind='bar', figsize=(10,6))
plt.subplots_adjust(bottom=0.5)
#plt.show()
print(airport_delays.sort_values(by='DepDelay', ascending=False))

# 3.e) Do late departures tend to result in longer arrival delays than on-time departures?
late_dep_arr_mean = flights_df.groupby('DepDel15')['ArrDelay'].mean().reset_index().at[1,'ArrDelay']
ontime_dep_arr_mean = flights_df.groupby('DepDel15')['ArrDelay'].mean().reset_index().at[0,'ArrDelay']
print("average arrival delay, when dep on time: ", ontime_dep_arr_mean)
print("average arrival delay, when dep late: ", late_dep_arr_mean)
arr_del_by_dep_del = pd.DataFrame(data = flights_df.groupby('DepDelay')['ArrDelay'].mean().reset_index())
arr_del_by_dep_del.plot(x='DepDelay', y='ArrDelay', kind='line', figsize=(10,7))
plt.show()

# 3.f) Which route (from origin airport to destination airport) has the most late arrivals?

flights_df['Route'] = flights_df.OriginAirportName + ' - ' + flights_df.DestAirportName
routes_df = pd.DataFrame(data = flights_df.groupby('Route')['ArrDel15'].sum().reset_index()) 

route_count = pd.DataFrame(data = flights_df.groupby('Route')['ArrDel15'].size().reset_index())
routes_df = pd.concat([routes_df, route_count.ArrDel15.rename('FlightCount')], axis = 1)
routes_df['PercOfDelays'] = round((routes_df.ArrDel15/routes_df.FlightCount)*100,2)
print(routes_df.sort_values(by='PercOfDelays', ascending=False))

# 3.g) Which route has the highest average arrival delay?
routes_agv_arr_del_df = pd.DataFrame(data = flights_df.groupby('Route')['ArrDelay'].mean().reset_index())
routes_df = pd.concat([routes_df, routes_agv_arr_del_df.ArrDelay.rename('AvgArrDel')], axis = 1)
routes_df = routes_df.sort_values(by='ArrDel15', ascending=False)
print(routes_df)
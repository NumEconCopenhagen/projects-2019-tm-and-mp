import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm

# Import CSV file with data from nordpoolgroup.com
df = pd.read_csv('DA_prices.csv', sep=';', decimal=',', thousands='.')
print(df)

df = df.rename(columns = {'WP Penetration (MWh)':'WP Penetration'})
df.loc[:, 'WP Penetration'] *= 100
print(df)

# Drop all rows and columns with only NaN values
df = df.dropna(axis=1, how='all')
df = df.dropna(axis=0, how='all')
print(df)

# OLS regression of WP Penetration on Day-Ahead price
regression = sm.OLS(df['Day-Ahead price (DKK)'], sm.add_constant(df['WP Penetration'])).fit()
regression.summary()
parameters = np.array(regression.params)
print(type(parameters))

# Calculating trendline
trend = pd.DataFrame({'trend':parameters[0] + parameters[1]*df['WP Penetration']})
trend = trend.round(2)
print(trend)

df = pd.concat([df, trend], axis=1)
print(df)

# Scatter plot and trendline from OLS regression
fig = plt.figure(dpi=100)
ax = fig.add_subplot(1,1,1)
ax.scatter(df['WP Penetration'], df['Day-Ahead price (DKK)'])
ax.plot(df['WP Penetration'], df['trend'], lw=2, c='black')
ax.set_xlabel('WP Penetration (%)')
ax.set_ylabel('Day-Ahead price (DKK)')
plt.show(ax)

# Creating a dummy variable for the winter, months and hour
df['month'] = df['HourDK'].str.slice(3,5)
df['month'] = pd.to_numeric(df['month'], downcast='integer')
winter = pd.Series([1, 2, 12])
df['winter'] = df['month'].isin(winter).astype(np.int8)
df['hour'] = df['HourDK'].str.slice(9,11)
df['hour'] = pd.to_numeric(df['hour'], downcast='integer')

# Create table with means for each variable for each month
table_group = df.groupby(['month'])['WP Prognosis (MWh)', 'Cons. Prognosis (MWh)', 'WP Penetration', 'Day-Ahead price (DKK)'].mean()
table_group.round(2)
table_group.to_csv('table_group.csv', sep=';', decimal=',')
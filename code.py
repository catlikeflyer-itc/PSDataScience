"""
Write all the code here 
"""
#%% imports (they work fine)
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import sys 
import pandas_profiling
sns.set()
from sympy import latex

#%% Create dataframe
nut = pd.read_csv('nutrition.csv', index_col=1) #nut :)
# Sirve pero tarda un vergo, por eso lo comento nut.profile_report()
print(f'Dataframe of size {nut.shape[0]} and {nut.shape[1]}')

#%% Cleaned dF
newnut = nut.dropna()
newnut2 = newnut[['calories','total_fat','carbohydrate','sodium','protein']]
newnut2

#%% Convert mg to g
newnut2['sodium'] = newnut2['sodium'].str.rstrip('mg').astype('float')
newnut2['total_fat'] = newnut2['total_fat'].str.rstrip('g').astype('float')
newnut2['carbohydrate'] = newnut2['carbohydrate'].str.rstrip('g').astype('float')
newnut2['protein'] = newnut2['protein'].str.rstrip('g').astype('float')

newnut2['sodium'] = newnut2['sodium']/1000
newnut2
#%%check shit on the .csv
nut_nuts = newnut2.loc[['Nuts, pecans']]
nut_ramen = newnut2.loc[['Soup, dry, beef flavor, ramen noodle']]
nut_teff = newnut2.loc[['Teff, uncooked']]
nut_sherbet = newnut2.loc[['Sherbet, orange']]

newnut3 = newnut2.loc[['Nuts, pecans','Soup, dry, beef flavor, ramen noodle','Teff, uncooked','Sherbet, orange']]

#%% histogram counts 
fig, axes = plt.subplots(3, 2, figsize=(20, 20), sharex=False)

fig.suptitle('Plot of 4', y=0.9, fontsize=20)

sns.histplot(newnut2["calories"], color="skyblue", stat='count', bins=100, ax=axes[0, 0])
sns.histplot(newnut2["total_fat"], color="olive", stat='count', bins=100, ax=axes[0, 1])
sns.histplot(newnut2["carbohydrate"], color="gold", stat='count', bins=100, ax=axes[1, 0])
sns.histplot(newnut2["sodium"], color="teal", stat='count', bins=100,ax=axes[1, 1])
sns.histplot(newnut2["protein"], color="blue", stat='count', bins=100,ax=axes[2, 0])
fig.delaxes(axes[2,1])
# nightmare nightmare nightmare nightmare nightmare nightmare nightmare nightmare nightmare nightmare nightmare nightmare nightmare nightmare nightmare

# %% Boxplot
f, axes = plt.subplots(nrows=3, ncols=2, figsize=(26,18))

f.suptitle('Boxplots of Fat, Sodium, Protein, Carbohydrates and Calories', y=0.9, fontsize=20)
sns.boxplot(x=newnut2['calories'], color="skyblue", ax=axes[0, 0])
sns.boxplot(x=newnut2['total_fat'], color="olive", ax=axes[0, 1])
sns.boxplot(x=newnut2['carbohydrate'], color="gold", ax=axes[1, 0])
sns.boxplot(x=newnut2['sodium'], color="teal", ax=axes[1, 1])
sns.boxplot(x=newnut2['protein'], color="blue", ax=axes[2, 0])

f.delaxes(axes[2,1])

#%% Filters outliers
Q1 = newnut2.quantile(0.25)
Q3 = newnut2.quantile(0.75)
IQR = Q3 - Q1

finalnut = newnut2[~((newnut2 < (Q1 - 2.5 * IQR)) |(10000 > (Q3 + 2.5 * IQR))).any(axis=1)]

# -------------------------------------------------------------------------------------------------

len_after = len(finalnut)
len_before = len(newnut2)
len_difference = len(newnut2) - len(finalnut)
print('We reduced our data size from {} foods by {} foods to {} foods.'.format(len_before, len_difference, len_after))

#%% Linear fit
newnut3 = newnut2

from sklearn.linear_model import LinearRegression

X = newnut3[['total_fat', 'carbohydrate', 'sodium', 'protein']] 
Y = newnut3['calories']

# with sklearn
regressionname = LinearRegression()
regressionname.fit(X, Y)

print('Intercept: \n', regressionname.intercept_)
print('Coefficients: \n', regressionname.coef_)
# %%
sys.exit()
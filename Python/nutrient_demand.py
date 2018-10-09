### Rural Kenya Nutrient Demand Analysis ###

import numpy as np
import pandas as pd
import glob
import statsmodels as sm
from scipy import stats


### Data management ###

def files(sub_dir):
    return glob.glob('/home/akappes/' + sub_dir + '*.csv')

nutrient_dfs = np.array(files('/Research/Africa/Nutrient_Demand/'))
print(nutrient_dfs)

df_livinc = pd.read_csv(nutrient_dfs[0])
df_consump = pd.read_csv(nutrient_dfs[1])
df_hh_demographs = pd.read_csv(nutrient_dfs[2])
df_nutrient_props = pd.read_csv(nutrient_dfs[3])
df_land_crop = pd.read_csv(nutrient_dfs[4])
df_hh_asset = pd.read_csv(nutrient_dfs[5])

# Need to match HH_id by month due to repeating/missing id values across months
# then concatenate all months Feb 13 - Jul 16

m_y = pd.DataFrame(df_consump['date'].unique().tolist()).rename(columns={0: 'date'})

d={}
for month_year in m_y['date']:
    d[month_year] = pd.DataFrame()

for key in d:

    d[key] = df_consump[df_consump['date'] == key].drop_duplicates(['HousehldID'], keep='last').merge(

        df_hh_asset[df_hh_asset['date'] == key].drop_duplicates(['HousehldID'], keep='last'),
        how='inner', on='HousehldID').merge(

        df_hh_demographs[df_hh_demographs['date'] == key].drop_duplicates(['HousehldID'], keep='last'),
        how='inner', on='HousehldID').merge(

        df_land_crop[df_land_crop['date'] == key].drop_duplicates(['HousehldID'], keep='last'),
        how='inner', on='HousehldID').merge(

        df_livinc[df_livinc['date'] == key].drop_duplicates(['HousehldID'], keep='last'),
        how='inner', on='HousehldID').dropna()


df_list = [d[m_y.loc[0, 'date']]]
i = 1
while i < len(d):
    df_list.append(d[m_y.loc[i, 'date']])
    i += 1
    if i > len(d):
        break

df = pd.concat(df_list, axis=0) # Ending with 15326

# Write algorithm to convert food consumption values to macronutrient proportions
# using USDA Food Composition Databases https://ndb.nal.usda.gov/ndb/



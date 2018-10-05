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

# Need to match HH_id by month due to repeating/missing id values across years

m_y = df_consump['date'].unique().tolist()

d={}
for month_year in m_y:
    d[month_year] = pd.DataFrame()

d[m_y[0]] = pd.merge(df_consump[df_consump['date'] == m_y[0]],
                     df_hh_asset[df_hh_asset['date'] == m_y[0]],
                     how='left', on='HousehldID')

# first month merger successful across two dfs, will write a loop for all other dfs

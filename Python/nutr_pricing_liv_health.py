### livestock health impacts on nutrient shadow pricing ###

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
import glob

def get_files(subpath):
    return glob.glob('/home/ajkappes/research/africa/' + subpath)

health_dfs = np.array(get_files('Livestock_Human_Health/data/*.csv'))
nutrient_dfs = np.array(get_files('Nutrient_Demand/*.csv'))

healthcare_df = pd.read_csv(health_dfs[0])
liv_gen_health_df = pd.read_csv(health_dfs[1])
liv_health_df = pd.read_csv(health_dfs[2])

for i in range(len(nutrient_dfs)):
    if 'pricing' in nutrient_dfs[i]:
        nutr_pricing_df = pd.read_csv(nutrient_dfs[i])
        print('Nutrient pricing data read at position', i)
        print()

month_year = pd.DataFrame(nutr_pricing_df['date'].unique().tolist()).rename(columns={0: 'date'})

d = {}
for m_y in month_year['date']:
    d[m_y] = pd.DataFrame()

for key in d:
    d[key] = nutr_pricing_df[nutr_pricing_df['date'] == key].drop_duplicates(['HousehldID'], keep='last').merge(

        healthcare_df[healthcare_df['date'] == key].drop_duplicates(['HousehldID'], keep='last'),
        how='inner', on='HousehldID').merge(

        liv_gen_health_df[liv_gen_health_df['date'] == key].drop_duplicates(['HousehldID'], keep='last'),
        how='inner', on='HousehldID')

def remove_no_obs(data_dict):
    no_obs = []
    for key in data_dict:
        if len(data_dict[key]) == 0:
            no_obs.append(key)

    for key in no_obs:
        del data_dict[key]
        print(key, 'has been removed due to no entries')

print('Entries removed from dictionary d')
remove_no_obs(d)
print()

def data_combine(data_dict):
    key_list = [key for key in data_dict]
    data_list = []
    for key in key_list:
        data_list.append(data_dict[key])

    return pd.concat(data_list, axis=0).reset_index().drop(columns='index')

# aggregate data construction
df = data_combine(d)

# micro livestock health data construnction
bovine_health = liv_health_df[liv_health_df['Species'] == 'BO']
goat_health = liv_health_df[liv_health_df['Species'] == 'OV']
sheep_health = liv_health_df[liv_health_df['Species'] == 'CP']

# merge livestock health data with aggregate data
d_bov_health = {}
d_goat_health = {}
d_sheep_health = {}
for m_y in month_year['date']:
    d_bov_health[m_y] = pd.DataFrame()
    d_goat_health[m_y] = pd.DataFrame()
    d_sheep_health[m_y] = pd.DataFrame()

for key in d_bov_health:
    d_bov_health[key] = df[df['date'] == key].drop_duplicates(['HousehldID'], keep='last').merge(
        bovine_health[bovine_health['date'] == key].drop_duplicates(['HousehldID'], keep='last'),
        how='inner', on='HousehldID')

    d_goat_health[key] = df[df['date'] == key].drop_duplicates(['HousehldID'], keep='last').merge(
        goat_health[goat_health['date'] == key].drop_duplicates(['HousehldID'], keep='last'),
        how='inner', on='HousehldID')

    d_sheep_health[key] = df[df['date'] == key].drop_duplicates(['HousehldID'], keep='last').merge(
        sheep_health[sheep_health['date'] == key].drop_duplicates(['HousehldID'], keep='last'),
        how='inner', on='HousehldID')

species_dict_list = [d_bov_health, d_goat_health, d_sheep_health]
species_list = ['bovine', 'goat', 'sheep']

for i in range(len(species_dict_list)) and range(len(species_list)):
    print('Entries removed from', species_list[i], 'data')
    remove_no_obs(species_dict_list[i])
    print()

df_p_bovine = data_combine(d_bov_health)
df_p_bovine.to_csv('/home/ajkappes/research/africa/Nutrient_Demand/np_bovine_health.csv')
df_p_goat = data_combine(d_goat_health)
df_p_goat.to_csv('/home/ajkappes/research/africa/Nutrient_Demand/np_goat_health.csv')
df_p_sheep = data_combine(d_sheep_health)
df_p_sheep.to_csv('/home/ajkappes/research/africa/Nutrient_Demand/np_sheep_health.csv')

### livestock health impacts on nutrient shadow pricing ###

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import scipy.stats
import glob


#### data management ####
#########################

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

    if 'cpi' in nutrient_dfs[i]:
        cpi_df = pd.read_csv(nutrient_dfs[i])
        print('CPI data read at position', i)
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
df_p_goat = data_combine(d_goat_health)
df_p_sheep = data_combine(d_sheep_health)

# df_p_bovine.to_csv('/home/ajkappes/research/africa/Nutrient_Demand/np_bovine_health.csv')
# df_p_goat.to_csv('/home/ajkappes/research/africa/Nutrient_Demand/np_goat_health.csv')
# df_p_sheep.to_csv('/home/ajkappes/research/africa/Nutrient_Demand/np_sheep_health.csv')

nutr_p_list = ['protein_p', 'fat_p', 'carb_p']

def defl_n_prices(df_species):
    df_key_list = df_species.iloc[:, 0].unique().tolist()
    df_defl_p = pd.DataFrame(columns=[p + '_defl' for p in nutr_p_list], index=df_species.index, dtype='float64')

    for i in range(len(nutr_p_list)):
        for key in df_key_list:
            df_defl_p.iloc[df_species[df_species.iloc[:, 0] == key].index.tolist(), i] = \
                df_species[df_species.iloc[:, 0] == key][nutr_p_list[i]] * float(cpi_df[cpi_df['date'] == key]['defl'])

    return pd.concat([df_species, df_defl_p], axis=1)

df_p_bovine = defl_n_prices(df_p_bovine)
df_p_goat = defl_n_prices(df_p_goat)
df_p_sheep = defl_n_prices(df_p_sheep)


def norm_price(df_species):
    norm_price_list = ['norm_' + p for p in nutr_p_list]
    defl_p_list = [p for p in df_species.columns if '_defl' in p]

    for i in range(len(norm_price_list)):
        df_species[norm_price_list[i]] = (df_species[defl_p_list[i]] - np.min(df_species[defl_p_list[i]])) / \
                                         (np.max(df_species[defl_p_list[i]]) - np.min(df_species[defl_p_list[i]]))

norm_price(df_p_bovine)
norm_price(df_p_goat)
norm_price(df_p_sheep)

#### livestock health impact on nutrient prices ####
####################################################

def get_y(df_species):
    return df_species[[p for p in df_species.columns if '_defl' in p]]

def get_X(df_species):
    return df_species[[d for d in df_species.columns if 'Disorders' in d or 'Illness' in d]]

y_bovine, X_bovine = get_y(df_p_bovine), get_X(df_p_bovine)
y_goat, X_goat = get_y(df_p_goat), get_X(df_p_goat)
y_sheep, X_sheep = get_y(df_p_sheep), get_X(df_p_sheep)

# check to see if non-values are entered in observations
for i in X_bovine.columns:
    print('Bovine ' + i, X_bovine[i].unique())
print()
for i in X_goat.columns:
    print('Goat ' + i, X_goat[i].unique())
print()
for i in X_sheep.columns:
    print('Sheep ' + i, X_sheep[i].unique())
print()

# for muscle, skin, and nerve disorders, change '77' for 'don't know' to '0' for 'no'
for i in X_bovine.columns:
    if 77 in X_bovine[i].unique():
        print('Bovine', i, 'value changed from 77 to 0 at index', X_bovine[X_bovine[i] == 77].index.tolist())
        X_bovine[i].replace(77, 0, inplace=True)
print()
for i in X_goat.columns:
    if 77 in X_goat[i].unique():
        print('Goat', i, 'value changed from 77 to 0 at index', X_goat[X_goat[i] == 77].index.tolist())
        X_goat[i].replace(77, 0, inplace=True)
print()
for i in X_sheep.columns:
    if 77 in X_sheep[i].unique():
        print('Sheep', i, 'value changed from 77 to 0 at index', X_sheep[X_sheep[i] == 77].index.tolist())
        X_sheep[i].replace(77, 0, inplace=True)
print()

def fe_mod(y, X):
    '''
    Performs linear regression for livestock health fixed effects on nutrient shadow prices. Dependent variables
    include protein, fat, and carb shadow prices. Model results are stored in list format accessible by index.

    Parameters: endogenous array, y
                exogenous array, X
    '''

    cols = y_bovine.columns
    model_list = []

    for i in cols:
        mod = sm.OLS(y[i], sm.add_constant(X)).fit()
        model_list.append(mod)

    return model_list

bovine_lm = fe_mod(y_bovine, X_bovine)
goat_lm = fe_mod(y_goat, X_goat)
sheep_lm = fe_mod(y_sheep, X_sheep)

# logistic construction and analysis for odds ratio impacts of livestock health on nutrient shadow prices

def get_bin_p(y_species):
    bin_p_list = ['bin_' + p for p in nutr_p_list]
    for i in range(len(bin_p_list)):
        print('Observation', y_species.columns[i], 'nutrient price values >=',
              np.mean(y_species[y_species.columns[i]]), 'coded as 1 with values less than the mean coded as 0')

        y_species[bin_p_list[i]] = np.where(y_species[y_species.columns[i]] >= np.mean(y_species[y_species.columns[i]]),
                                           1, 0)
Y = [y_bovine, y_goat, y_sheep]
for s, y in zip(species_list, Y):
    print(s + ':')
    get_bin_p(y)
    print()

def get_logit_results(y_species, X_species):
    '''
    Performs logistic regression to evaluate effects of livestock general illness on nutrient prices greater than or
     equal to its mean or less than its mean.

    Returns a list with elements consisting of intercept and general illness coefs, odds ratio, probability
     of nutrient price being greater than the mean given general livestock illness occurs, partial effects of
     general livestock illness given nutrient prices greater than the mean.

    Nutrient list positions:
    0 - protein shadow price
    1 - fat shadow price
    2 - carb shadow price
    '''

    y_bin_list = [p for p in y_species.columns if 'bin_' in p]
    params = []

    for i in y_bin_list:
        lr = LogisticRegression(solver='lbfgs').fit(np.array(X_species[X_species.columns[0]]).reshape(-1, 1),
                                                    y_species[i])

        prob = 1 - stats.norm.cdf(float(lr.intercept_ + lr.coef_))
        partial_effect = prob - (1 - stats.norm.cdf(float(lr.intercept_)))
        results = [float(lr.intercept_), float(lr.coef_), np.exp(float(lr.coef_)),
                   prob, partial_effect]

        params.append(results)

    return params


bovine_lr = get_logit_results(y_bovine, X_bovine)
goat_lr = get_logit_results(y_goat, X_goat)
sheep_lr = get_logit_results(y_sheep, X_sheep)



# Partial effect of general livestock illness on nutrient price classification for price = 1 (above mean):
# Prob(np > mean np | general livestock illness) - Prob(np > mean np | no general livestock illness)

# reveals a probability the nutrient price is greater when general livestock
# illness is present than when illness is not present, given nutrient price is 1 (above mean)


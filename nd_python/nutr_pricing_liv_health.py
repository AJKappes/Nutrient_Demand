### livestock health impacts on nutrient shadow pricing ###

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import scipy.stats
import glob
plt.style.use('ggplot')

#### data management ####
#########################

def get_files(subpath):
    return glob.glob('/home/alex/research/africa/' + subpath)

health_dfs = np.array(get_files('Livestock_Human_Health/data/*.csv'))
nutrient_dfs = np.array(get_files('Nutrient_Demand/*.csv'))

liv_gen_health_df = pd.read_csv(health_dfs[0])
healthcare_df = pd.read_csv(health_dfs[1])
liv_health_df = pd.read_csv(health_dfs[2])


for i in range(len(nutrient_dfs)):
    if 'pricing' in nutrient_dfs[i]:
        nutr_pricing_df = pd.read_csv(nutrient_dfs[i])
        print('Nutrient pricing data read at position', i)

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

# micro livestock health data construction
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

# df_p_bovine.to_csv('/home/alex/research/africa/Nutrient_Demand/np_bovine_health.csv')
# df_p_goat.to_csv('/home/aalex/research/africa/Nutrient_Demand/np_goat_health.csv')
# df_p_sheep.to_csv('/home/alex/research/africa/Nutrient_Demand/np_sheep_health.csv')

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

# fixed effect model for livestock health impacts on nutrient shadow prices
def fe_mod(y, X, cov_spec=False):
    '''
    Performs linear regression for livestock health fixed effects on nutrient shadow prices. Dependent variables
    include protein, fat, and carb shadow prices. Model results are stored in list format accessible by index.

    Parameters: endogenous array, y
                exogenous array, X
    '''

    cols = y_bovine.columns
    model_list = []

    for i in cols:
        if cov_spec == 'HAC':
            mod = sm.OLS(y[i], sm.add_constant(X)).fit(cov_type='HAC', cov_kwds={'maxlags': 1})
            model_list.append(mod)
        else:
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
              round(np.mean(y_species[y_species.columns[i]]), 3),
              'coded as 1 with values less than the mean coded as 0')

        y_species[bin_p_list[i]] = np.where(y_species[y_species.columns[i]] >=
                                            np.mean(y_species[y_species.columns[i]]), 1, 0)
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

        # probability nutrient price is above mean given general livestock illness
        prob = stats.norm.cdf(float(lr.intercept_ + lr.coef_))
        # partial effect between healthy and sick ivestock on probability of nutrient price being above the mean
        partial_effect = stats.norm.cdf(float(lr.intercept_)) - prob
        results = [float(lr.intercept_), float(lr.coef_), np.exp(float(lr.coef_)),
                   prob, partial_effect]

        params.append(results)

    return params


bovine_lr = get_logit_results(y_bovine, X_bovine)
goat_lr = get_logit_results(y_goat, X_goat)
sheep_lr = get_logit_results(y_sheep, X_sheep)

df_lr_results = pd.DataFrame()
df_lr_results['Species'] = np.array(['Bovine', '', '',
                                     'Goat', '', '',
                                     'Sheep', '', ''])
df_lr_results['Nutrient'] = np.array(['Protein', 'Fat', 'Carbohydrate'] * 3)

odds_list = []
prob_list = []
pe_list = []
lr_list = [bovine_lr, goat_lr, sheep_lr]
for species in lr_list:
    for i in range(len(species)):
        odds_list.append(species[i][2])
        prob_list.append(species[i][3])
        pe_list.append(species[i][4])

df_lr_results['Odds_Ratio'] = np.round(odds_list, 4)
df_lr_results['Probability'] = np.round(prob_list, 4)
df_lr_results['Partial_Effect_Healthy'] = np.round(pe_list, 4)

# Partial effect of general livestock illness on nutrient price classification for price = 1 (above mean):
# Prob(np > mean np | general livestock illness) - Prob(np > mean np | no general livestock illness)

# reveals a probability the nutrient price is greater when general livestock
# illness is present than when illness is not present, given nutrient price is 1 (above mean)

# livestock illness instrumental variable approach

df_list = [df_p_bovine, df_p_goat, df_p_sheep]
X_list = [X_bovine, X_goat, X_sheep]

for d, x in zip(df_list, X_list):
    d[[j for j in d.columns if 'Disorders' in j]] = x[[j for j in x.columns if 'Disorders' in j]]

# IV illness aggregation

for data in df_list:
    ill_list = [j for j in data.columns if 'Disorders' in j]
    data['ill_sum'] = data[ill_list].sum(axis=1)

# IV village illness aggregation

vill_list = df_p_bovine['VillageID'].unique().tolist()

bov_my = df_p_bovine.iloc[:, 0].unique().tolist()
goat_my = df_p_goat.iloc[:, 0].unique().tolist()
sheep_my = df_p_sheep.iloc[:, 0].unique().tolist()
species_my = [bov_my, goat_my, sheep_my]

bov_ill_avg = np.zeros(shape=len(df_p_bovine))
goat_ill_avg = np.zeros(shape=len(df_p_goat))
sheep_ill_avg = np.zeros(shape=len(df_p_sheep))
ill_avg = [bov_ill_avg, goat_ill_avg, sheep_ill_avg]

i = 0
for d, ill in zip(df_list, ill_avg):
    for date in species_my[i]:
        idx_date = d[d.iloc[:, 0] == date].index.tolist()
        vill_list = d.loc[idx_date, 'VillageID'].unique().tolist()

        for vil in vill_list:
            dv_df = d.loc[idx_date, [c for c in d.columns if 'Disorders' in c or 'Village' in c]]
            dv_df = dv_df.loc[dv_df['VillageID'] == vil, [c for c in dv_df.columns if 'Disorders' in c]]
            ill[dv_df.index.tolist()] = sum(dv_df.sum()) / len(dv_df)

    d['ill_avg'] = ill
    i += 1

# add additional features to livestock disease IV

nd_data_list = list(get_files('Nutrient_Demand/*.csv'))
for i in range(len(nd_data_list)):
    if 'asset' in nd_data_list[i]:
        asset_idx = i
        print('asset df located at index', i)
        hh_asset_df = pd.read_csv(nd_data_list[i])
    elif 'demographs' in nd_data_list[i]:
        dem_idx = i
        print('demograph df located at index', i)
        dem_df = pd.read_csv(nd_data_list[i])

# merge asset and demograph features

d_bov = {}
d_goat = {}
d_sheep = {}
d_list = [d_bov, d_goat, d_sheep]
for i in range(len(d_list)):
    d_list[i] = pd.DataFrame()

# for i in range(len(d_list)):
#     for key in species_my[i]:
#         print(key)
#         d_list[i][key] = df_list[i][df_list[i].iloc[:, 0] == key].drop_duplicates(['HousehldID'], keep='last').merge(
#             hh_asset_df[hh_asset_df['date'] == key].drop_duplicates(['HousehldID'], keep='last'),
#             how='inner', on='HousehldID').merge(
#
#             dem_df[dem_df['date'] == key].drop_duplicates(['HousehldID'], keep='last'), how='inner', on='HousehldID')

for key in species_my[0]:
    d_bov[key] = df_p_bovine[df_p_bovine.iloc[:, 0] == key].drop_duplicates(['HousehldID'], keep='last').merge(
        hh_asset_df[hh_asset_df['date'] == key].drop_duplicates(['HousehldID'], keep='last'),
        how='inner', on='HousehldID').merge(

        dem_df[dem_df['date'] == key].drop_duplicates(['HousehldID'], keep='last'), how='inner', on='HousehldID')

for key in species_my[1]:
    d_goat[key] = df_p_goat[df_p_goat.iloc[:, 0] == key].drop_duplicates(['HousehldID'], keep='last').merge(
        hh_asset_df[hh_asset_df['date'] == key].drop_duplicates(['HousehldID'], keep='last'),
        how='inner', on='HousehldID').merge(

        dem_df[dem_df['date'] == key].drop_duplicates(['HousehldID'], keep='last'), how='inner', on='HousehldID')

for key in species_my[2]:
    d_sheep[key] = df_p_sheep[df_p_sheep.iloc[:, 0] == key].drop_duplicates(['HousehldID'], keep='last').merge(
        hh_asset_df[hh_asset_df['date'] == key].drop_duplicates(['HousehldID'], keep='last'),
        how='inner', on='HousehldID').merge(

        dem_df[dem_df['date'] == key].drop_duplicates(['HousehldID'], keep='last'), how='inner', on='HousehldID')

df_p_bovine = data_combine(d_bov)
df_p_goat = data_combine(d_goat)
df_p_sheep = data_combine(d_sheep)

iv_sum_vars = ['ill_sum', 'TotalHHMembers']
iv_avg_vars = ['ill_avg', 'TotalHHMembers']
X_sum_bovine = sm.add_constant(df_p_bovine[iv_sum_vars])
X_avg_bovine = sm.add_constant(df_p_bovine[iv_avg_vars])
X_sum_goat = sm.add_constant(df_p_goat[iv_sum_vars])
X_avg_goat = sm.add_constant(df_p_goat[iv_avg_vars])
X_sum_sheep = sm.add_constant(df_p_sheep[iv_sum_vars])
X_avg_sheep = sm.add_constant(df_p_sheep[iv_avg_vars])

iv_sum_bov_mod = fe_mod(y_bovine, X_sum_bovine, cov_spec='HAC')
iv_avg_bov_mod = fe_mod(y_bovine, X_avg_bovine, cov_spec='HAC')
iv_sum_goat_mod = fe_mod(y_goat, X_sum_goat, cov_spec='HAC')
iv_avg_goat_mod = fe_mod(y_goat, X_avg_goat, cov_spec='HAC')
iv_sum_sheep_mod = fe_mod(y_sheep, X_sum_sheep, cov_spec='HAC')
iv_avg_sheep_mod = fe_mod(y_sheep, X_avg_sheep, cov_spec='HAC')
iv_sum_mods = [iv_sum_bov_mod, iv_sum_goat_mod, iv_sum_sheep_mod]
iv_avg_mods = [iv_avg_bov_mod, iv_avg_goat_mod, iv_avg_sheep_mod]


dep = ['protein', '', '',
       'fat', '', '',
       'carb', '', '']

var_sum_names = ['const', 'Livestock Illness Freq', 'Total HH Members'] * 3
var_avg_names = ['const', 'Livestock Illness Avg', 'Total HH Members'] * 3

iv_sum_results = []
iv_avg_results = []
for m in iv_sum_mods:
    params = np.concatenate([np.array(m[0].params),
                             np.array(m[1].params),
                             np.array(m[2].params)])

    se = np.concatenate([np.array(m[0].bse),
                         np.array(m[1].bse),
                         np.array(m[2].bse)])

    tvals = np.concatenate([np.array(m[0].tvalues),
                            np.array(m[1].tvalues),
                            np.array(m[2].tvalues)])

    pvals = np.concatenate([np.array(m[0].pvalues),
                            np.array(m[1].pvalues),
                            np.array(m[2].pvalues)])

    iv_sum_results.append(round(pd.DataFrame({'Dependent': dep,
                                          'Independent': var_sum_names,
                                          'Coef': params,
                                          'Std Errors': se,
                                          't-values': tvals,
                                          'p-values': pvals}), 4))

for m in iv_avg_mods:
    params = np.concatenate([np.array(m[0].params),
                             np.array(m[1].params),
                             np.array(m[2].params)])

    se = np.concatenate([np.array(m[0].bse),
                         np.array(m[1].bse),
                         np.array(m[2].bse)])

    tvals = np.concatenate([np.array(m[0].tvalues),
                            np.array(m[1].tvalues),
                            np.array(m[2].tvalues)])

    pvals = np.concatenate([np.array(m[0].pvalues),
                            np.array(m[1].pvalues),
                            np.array(m[2].pvalues)])

    iv_avg_results.append(round(pd.DataFrame({'Dependent': dep,
                                          'Independent': var_avg_names,
                                          'Coef': params,
                                          'Std Errors': se,
                                          't-values': tvals,
                                          'p-values': pvals}), 4))


### tables and plots ###

# variable summary stats
vars = ['protein_prop', 'fat_prop', 'carb_prop',
        'protein_p_defl', 'fat_p_defl', 'carb_p_defl',
        'ill_avg', 'TotalHHMembers']

var_df = pd.concat([df_p_bovine[vars], df_p_goat[vars], df_p_sheep[vars]])

vars_ss = pd.DataFrame(var_df.describe()).rename(columns={'protein_prop': 'Protein',
                                                          'fat_prop': 'Lipids',
                                                          'carb_prop': 'Carbohydrates',
                                                          'protein_p_defl': 'Protein',
                                                          'fat_p_defl': 'Lipids',
                                                          'carb_p_defl': 'Carbohydrates',
                                                          'ill_avg': 'Livestock Illness Village Average',
                                                          'TotalHHMembers': 'Total Household Members'})

vars_ss = round(vars_ss[vars_ss.index.isin(['count', 'mean', 'std', 'min', 'max'])], 4)

# tables

print(vars_ss.to_latex())
print()

print('IV: Livestock Illness Frequency')
print()
for specie, results in zip(species_list, iv_sum_results):
    print(specie + ':')
    print(results.to_latex())
    print()

print('IV: Livestock Illness Village-Time Period Avg')
print()
for specie, results in zip(species_list, iv_avg_results):
    print(specie + ':')
    print(results.to_latex())
    print()

# deflated shadow price and consumption plots

df = defl_n_prices(df)
df_date_list = df['date_x'].unique().tolist()
mean_defl_p = []
mean_prop_p = []
mean_defl_f = []
mean_prop_f = []
mean_defl_c = []
mean_prop_c = []


for m_y in df_date_list:
    mean_defl_p.append(df[df.iloc[:, 0] == m_y]['protein_p_defl'].mean())
    mean_prop_p.append(df[df.iloc[:, 0] == m_y]['protein_prop'].mean())
    mean_defl_f.append(df[df.iloc[:, 0] == m_y]['fat_p_defl'].mean())
    mean_prop_f.append(df[df.iloc[:, 0] == m_y]['fat_prop'].mean())
    mean_defl_c.append(df[df.iloc[:, 0] == m_y]['carb_p_defl'].mean())
    mean_prop_c.append(df[df.iloc[:, 0] == m_y]['carb_prop'].mean())

mean_defl_list = [mean_defl_p, mean_defl_f, mean_defl_c]
mean_prop_list = [mean_prop_p, mean_prop_f, mean_prop_c]

# plt.rc('font', family='liberation serif')
# colors = ['b', 'g', 'r']
#
# plt.figure(figsize=(8, 5.5))
# for i in range(len(mean_defl_list)):
#     plt.plot(df_date_list, mean_defl_list[i], colors[i], alpha=0.4)
#
# plt.legend(('Protein', 'Lipids', 'Carbohydrates'), loc='upper right')
# plt.ylabel('Real Nutrient Shadow Price (Ksh/g)')
# plt.xticks(rotation=90)
#
# plt.figure(figsize=(8, 5.5))
# for i in range(len(mean_prop_list)):
#     plt.plot(df_date_list, mean_prop_list[i], colors[i], alpha=0.4)
#
# plt.legend(('Protein', 'Lipids', 'Carbohydrates'), loc='upper right')
# plt.ylabel('Nutrient Dietary Proportion (%)')
# plt.xticks(rotation=90)

for d,s in zip(df_list, species_list):
    print(s, ':')
    print()
    print('Illness:', len(d[d['GeneralIllness'] == 1]), 'No illness:', len(d[d['GeneralIllness'] == 0]))
    print()

defl_p_list = [j for j in df_p_bovine.columns if '_defl' in j]

specie_ill_avg = []
for d in df_list:
    specie_ill_avg.append(d['ill_avg'].mean())

for d, s, I in zip(df_list, species_list, specie_ill_avg):
    print(s + ':')
    print('(Illness)')
    print(round(d[d['ill_avg'] >= I][defl_p_list].mean(), 4))
    print()
    print('(No Illness)')
    print(round(d[d['ill_avg'] < I][defl_p_list].mean(), 4))
    print()

# avg deflated prices > ill specie avg
# shadow prices for >= ill specie avg > shadow prices for < ill specie avg

# get all rows where no disorders occur
disorders = [j for j in df_p_bovine.columns if 'Disorders' in j]

# evaluate shadow price difference for no disorder and disorder occurrence
for d, s in zip(df_list, species_list):
    idx = d[disorders].loc[(d[disorders] == 0).all(axis=1)].index.tolist()
    print(s, 'mean shadow prices' + ':')
    print()
    print('(no disorder occurrence)')
    print(d.loc[idx, defl_p_list].mean())
    print()
    print('(disorder occurrence)')
    print(d.loc[~(d.index.isin(idx)), defl_p_list].mean())
    print()



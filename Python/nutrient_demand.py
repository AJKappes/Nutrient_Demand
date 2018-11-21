### Rural Kenya Nutrient Demand Analysis ###

import numpy as np
import pandas as pd
import glob
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
from scipy import stats
import scipy.optimize as optim


### Data management ###

def files(sub_dir):
    return glob.glob('/home/ajkappes/' + sub_dir + '*.csv')

nutrient_dfs = np.array(files('/Research/Africa/Nutrient_Demand/'))
print(nutrient_dfs)

df_consump = pd.read_csv(nutrient_dfs[0])
df_nutrient_props = pd.read_csv(nutrient_dfs[1])
df_land_crop = pd.read_csv(nutrient_dfs[2])
df_hh_demographs = pd.read_csv(nutrient_dfs[3])
df_livinc = pd.read_csv(nutrient_dfs[4])
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

df = pd.concat(df_list, axis=0).reset_index().drop(columns='index') # Ending with 15326 observations

# Write algorithm to convert food consumption values to macronutrient proportions
# using USDA Food Composition Databases https://ndb.nal.usda.gov/ndb/

consump_l = ['TotalCowMilkConsumed', 'TotalGoatMilkConsumed', 'TotalEggsConsumed', 'TotalBeefConsumed',
             'TotalGoatMeatConsumed', 'TotalOtherMeatConsumed', 'TotalFishConsumed', 'TotalMaizeConsumed',
             'TotalCassavaConsumed', 'TotalSorghumConsumed', 'TotalBananaConsumed', 'TotalPulsesConsumed',
             'TotalViungoConsumed', 'TotalGreensConsumed', 'TotalPotatoConsumed', 'TotalOilConsumed']

consumption = df[consump_l]

# Converting '300ML' string in oil consumption to L float value and 2kg in pulses to 2
consumption.loc[consumption[consumption['TotalOilConsumed'] == '300ML'].index, 'TotalOilConsumed'] = 0.3
consumption.loc[consumption[consumption['TotalPulsesConsumed'] == '2kg'].index, 'TotalPulsesConsumed'] = 2
consumption = consumption.astype('float64')


macronutrients = ['protein', 'fat', 'carb']

goat_sheep_milk = pd.DataFrame(df_nutrient_props[df_nutrient_props['food_source'].isin(['milk_sheep', 'milk_goat'])]
                               [macronutrients].mean()).T

pulses = pd.DataFrame(df_nutrient_props[df_nutrient_props['food_source'].isin(['peas', 'navy', 'pinto',
                                                                               'black', 'kidney', 'lentils'])]
                      [macronutrients].mean()).T

viungo = pd.DataFrame(df_nutrient_props[df_nutrient_props['food_source'].isin(['onions', 'tomatoes', 'carrots',
                                                                               'green_peppers'])]
                      [macronutrients].mean()).T

greens = pd.DataFrame(df_nutrient_props[df_nutrient_props['food_source'].isin(['spinach', 'cabbage'])]
                      [macronutrients].mean()).T

nutrient_comps = pd.concat([df_nutrient_props[df_nutrient_props['food_source'] == 'milk_cow'][macronutrients],
                            goat_sheep_milk,
                            df_nutrient_props[df_nutrient_props['food_source'] == 'eggs'][macronutrients],
                            df_nutrient_props[df_nutrient_props['food_source'] == 'beef'][macronutrients],
                            df_nutrient_props[df_nutrient_props['food_source'] == 'goat'][macronutrients],
                            df_nutrient_props[df_nutrient_props['food_source'] == 'other_poultry'][macronutrients],
                            df_nutrient_props[df_nutrient_props['food_source'] == 'fish'][macronutrients],
                            df_nutrient_props[df_nutrient_props['food_source'] == 'maize'][macronutrients],
                            df_nutrient_props[df_nutrient_props['food_source'] == 'cassava'][macronutrients],
                            df_nutrient_props[df_nutrient_props['food_source'] == 'sorghum'][macronutrients],
                            df_nutrient_props[df_nutrient_props['food_source'] == 'banana'][macronutrients],
                            pulses, viungo, greens,
                            df_nutrient_props[df_nutrient_props['food_source'] == 'potato'][macronutrients],
                            df_nutrient_props[df_nutrient_props['food_source'] == 'oil'][macronutrients]],
                           axis=0).reset_index().drop(columns='index')

# Macronutrient proportions are values per 100g of food item consumed
# conversions: 1000g[water] per L
#              1000g per Kg
#
# measured in L: cow and goat/sheep milk
# measured in Kg: all meat and crop items
# eggs measured in # consumed: medium egg = 44g => scale 100g egg nutrient values by 0.44

conversion = np.array([10, 10, 0.44, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10])
consumption = np.array(consumption)
nutrient_comps = np.array(nutrient_comps)

macros = pd.DataFrame({'protein_cons': np.zeros(len(consumption)),
                       'fat_cons': np.zeros(len(consumption)),
                       'carbs_cons': np.zeros(len(consumption))})


for i in range(consumption.shape[0]):
    macros.loc[i, 'protein_cons'] = np.dot(np.multiply(nutrient_comps[:, 0], consumption[i]), conversion)
    macros.loc[i, 'fat_cons'] = np.dot(np.multiply(nutrient_comps[:, 1], consumption[i]), conversion)
    macros.loc[i, 'carbs_cons'] = np.dot(np.multiply(nutrient_comps[:, 2], consumption[i]), conversion)

# macronutrient monthly mean stats

macro_idx = {}
for month_year in m_y['date']:
    macro_idx[month_year] = pd.DataFrame()

for key in macro_idx:
    macro_idx[key] = df[df['date'] == key].index.tolist()

d_macros = {}
for month_year in m_y['date']:
    d_macros[month_year] = pd.DataFrame()

for key in d_macros:
    d_macros[key] = macros.loc[macro_idx[key]].mean()

df_macro_statlist = [d_macros[m_y.loc[0, 'date']]]
i = 1
while i < len(d_macros):
    df_macro_statlist.append(d_macros[m_y.loc[i, 'date']])
    i += 1
    if i > len(d_macros):
        break

df_macrostat = pd.DataFrame(pd.concat(df_macro_statlist, axis=1).T).round(3)
df_macrostat.index = m_y['date']

# macronutrient means plot
x_dates = df_macrostat.index

trace0 = go.Scatter(
    x=x_dates,
    y=df_macrostat['protein_cons'],
    mode='lines',
    name='Protein'
)

trace1 = go.Scatter(
    x=x_dates,
    y=df_macrostat['fat_cons'],
    mode='lines',
    name='Fat'
)

trace2 = go.Scatter(
    x=x_dates,
    y=df_macrostat['carbs_cons'],
    mode='lines',
    name='Carbohydrates'
)

line_data = [trace0, trace1, trace2]
layout = dict(title='Mean Macronutrient Consumption (7 day periods)',
              yaxis=dict(title='Macronutrient Consumption in Grams'))

figure = dict(data=line_data, layout=layout)
#plotly.offline.plot(figure, filename='Macronutrient_means_plot.html')

# shadow price construction
# each macronutrient's proportion of total food expense
food_exp_list = [var for var in df_consump.columns if 'Cost' in var][:-4] # removing non-food exps
df['total_fd_exp'] = df_consump[food_exp_list].sum(axis=1) # total cost of food across all foods consumed

macro_props = ['protein_prop', 'fat_prop', 'carb_prop']
for i in range(len(macro_props)):
    macros[macro_props[i]] = macros[macros.columns[i]] / macros[['protein_cons', 'fat_cons', 'carbs_cons']].sum(axis=1)

macro_cost = ['protein_cost', 'fat_cost', 'carb_cost']
for i in range(len(macro_cost)):
    macros[macro_cost[i]] = macros[macro_props[i]] * df['total_fd_exp']

macro_price = ['protein_p', 'fat_p', 'carb_p']
for i in range(len(macro_price)):
    macros[macro_price[i]] = df['total_fd_exp'] / macros[macros.columns[i]]

### demand estimation ###

# almost ideal demand system, deaton and muellbauer (1980)
# the following provides nonlinear estimation by assumed normal MLE

macros_sub = macros.dropna()

p_df = macros_sub[['protein_p', 'fat_p', 'carb_p']]
p_df = np.log(p_df.loc[~(p_df == 0).all(axis=1)])
p_df.columns = ['lnprotein_p', 'lnfat_p', 'lncarb_p']

def d_sum(var):
    return p_df[var] * p_df[['lnprotein_p', 'lnfat_p', 'lncarb_p']].sum(axis=1)

p_df['lnprotein_sum'] = d_sum('lnprotein_p')
p_df['lnfat_sum'] = d_sum('lnfat_p')
p_df['lncarb_sum'] = d_sum('lncarb_p')
p_df = p_df.replace([np.inf, -np.inf], np.nan).dropna()

X = np.concatenate([np.array(p_df[['lnprotein_p', 'lnfat_p', 'lncarb_p']]),
                    np.array(df.loc[p_df.index.tolist(), 'total_fd_exp']).reshape(len(p_df), 1),
                    np.array(p_df[['lnprotein_sum', 'lnfat_sum', 'lncarb_sum']])],
                   axis=1)

y = np.array(macros.loc[p_df.index.tolist(), ['protein_prop', 'fat_prop', 'carb_prop']])

# function specified by deaton and muellbauer (1980)
def fun_fit(p, X, y):
    # alpha params 0-3
    # beta param 4
    # gamma param 5-9
    return ((p[1] - p[4] * p[0]) + p[5] * X[:, 0] + p[6] * X[:, 1] + p[7] * X[:, 2] +
            p[4] * (X[:, 3] - (p[2] * X[:, 1] + p[3] * X[:, 2]) - 0.5 * (p[8] * X[:, 5] + p[9] * X[:, 6]))
            ) - y[:, 0]

def jacobian(p, X, y):
    j = np.empty((X.shape[0], p.size))
    j[:, 0] = -p[4]
    j[:, 1] = 1
    j[:, 2] = -p[4] * X[:, 1]
    j[:, 3] = -p[4] * X[:, 2]
    j[:, 4] = -p[0] + X[:, 3] - (p[2] * X[:, 1] + p[3] * X[:, 2]) - 0.5 * (p[8] * X[:, 5] + p[9] * X[:, 6])
    j[:, 5] = X[:, 0]
    j[:, 6] = X[:, 1]
    j[:, 7] = X[:, 2]
    j[:, 8] = p[4] * X[:, 5]
    j[:, 9] = p[4] * X[:, 6]
    return j

p_init = np.repeat(1, 10)
fit = optim.least_squares(fun_fit, p_init, jac=jacobian, args=(X, y), verbose=1)

# inference
n = X.shape[0]
k = p_init.size
sighat2 = (np.matrix(fit.fun) * np.matrix(fit.fun).T)[0, 0] / (n - k)
J = np.matrix(fit.jac)
param_cov = sighat2 * np.linalg.inv(J.T * J)
param_se = np.sqrt(np.diag(param_cov))
param_tvals = np.divide(fit.x, param_se)
t_crit = stats.t.ppf(.975, n - k)
param_pvals = 2 * (1 - stats.t.cdf(abs(param_tvals), n - k))
# own-price effect is only sig variable
# may be expected when analyzing at macronutrient level

# TODO compare to corrected stone index OLS method


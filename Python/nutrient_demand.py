### Rural Kenya Nutrient Demand Analysis ###

import numpy as np
import pandas as pd
import glob
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
from scipy import stats
import scipy.optimize as optim


###################### Data management ##################################
#########################################################################

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

df = pd.concat(df_list, axis=0).reset_index().drop(columns='index')  # Ending with 15326 observations

# Write algorithm to convert food consumption values to macronutrient proportions
# using USDA Food Composition Databases https://ndb.nal.usda.gov/ndb/

consump_l = ['TotalCowMilkConsumed', 'TotalGoatMilkConsumed', 'TotalEggsConsumed', 'TotalBeefConsumed',
             'TotalGoatMeatConsumed', 'TotalOtherMeatConsumed', 'TotalFishConsumed', 'TotalMaizeConsumed',
             'TotalCassavaConsumed', 'TotalSorghumConsumed', 'TotalBananaConsumed', 'TotalPulsesConsumed',
             'TotalViungoConsumed', 'TotalGreensConsumed', 'TotalPotatoConsumed', 'TotalOilConsumed']

consumption = df[consump_l]

# Converting '300ML' string in oil consumption to L float value and 2kg in pulses to 2
consumption.loc[consumption[consumption['TotalOilConsumed'] == '300ML'].index.tolist(), 'TotalOilConsumed'] = 0.3
consumption.loc[consumption[consumption['TotalPulsesConsumed'] == '2kg'].index.tolist(), 'TotalPulsesConsumed'] = 2
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

####################### demand estimation ###############################
#########################################################################

# almost ideal demand system, deaton and muellbauer (1980)
# the following provides nonlinear estimation by assumed normal MLE

macros = macros.replace([np.inf, -np.inf], np.nan).dropna()

p_df = macros[['protein_p', 'fat_p', 'carb_p']]
p_df = np.log(p_df.loc[~(p_df == 0).all(axis=1)])
p_df.columns = ['lnprotein_p', 'lnfat_p', 'lncarb_p']

def d_sum(var):
    return p_df[var] * p_df[['lnprotein_p', 'lnfat_p', 'lncarb_p']].sum(axis=1)

p_df['lnprotein_sum'] = d_sum('lnprotein_p')
p_df['lnfat_sum'] = d_sum('lnfat_p')
p_df['lncarb_sum'] = d_sum('lncarb_p')

X = np.concatenate([np.array(p_df[['lnprotein_p', 'lnfat_p', 'lncarb_p']]),
                    np.array(df.loc[p_df.index.tolist(), 'total_fd_exp']).reshape(len(p_df), 1),
                    np.array(p_df[['lnprotein_sum', 'lnfat_sum', 'lncarb_sum']])],
                   axis=1)

y = np.array(macros.loc[p_df.index.tolist(), ['protein_prop', 'fat_prop', 'carb_prop']])

# function specified by deaton and muellbauer (1980)

def fun_fit_aids(p, X, y, indicate):
    # alpha params 0-3
    # beta param 4
    # gamma param 5-9
    if indicate == 'protein':
        return ((p[1] - p[4] * p[0]) + p[5] * X[:, 0] + p[6] * X[:, 1] + p[7] * X[:, 2] +
                p[4] * (X[:, 3] - (p[2] * X[:, 1] + p[3] * X[:, 2]) - 0.5 * (p[8] * X[:, 5] + p[9] * X[:, 6]))
                ) - y[:, 0]

    elif indicate == 'fat':
        return ((p[1] - p[4] * p[0]) + p[5] * X[:, 0] + p[6] * X[:, 1] + p[7] * X[:, 2] +
                p[4] * (X[:, 3] - (p[2] * X[:, 0] + p[3] * X[:, 2]) - 0.5 * (p[8] * X[:, 4] + p[9] * X[:, 6]))
                ) - y[:, 1]

    else:
        return ((p[1] - p[4] * p[0]) + p[5] * X[:, 0] + p[6] * X[:, 1] + p[7] * X[:, 2] +
                p[4] * (X[:, 3] - (p[2] * X[:, 0] + p[3] * X[:, 1]) - 0.5 * (p[8] * X[:, 4] + p[9] * X[:, 5]))
                ) - y[:, 2]

def jacobian_aids(p, X, y, indicate):
    j = np.empty((X.shape[0], p.size))

    if indicate == 'protein':
        j[:, 2] = -p[4] * X[:, 1]
        j[:, 3] = -p[4] * X[:, 2]
        j[:, 8] = -p[4] * 0.5 * X[:, 5]
        j[:, 9] = -p[4] * 0.5 * X[:, 6]

    elif indicate == 'fat':
        j[:, 2] = -p[4] * X[:, 0]
        j[:, 3] = -p[4] * X[:, 2]
        j[:, 8] = -p[4] * 0.5 * X[:, 4]
        j[:, 9] = -p[4] * 0.5 * X[:, 6]

    else:
        j[:, 2] = -p[4] * X[:, 0]
        j[:, 3] = -p[4] * X[:, 1]
        j[:, 8] = -p[4] * 0.5 * X[:, 4]
        j[:, 9] = -p[4] * 0.5 * X[:, 5]

    j[:, 0] = -p[4]
    j[:, 1] = 1
    j[:, 4] = -p[0] + X[:, 3] - (p[2] * X[:, 1] + p[3] * X[:, 2]) - 0.5 * (p[8] * X[:, 5] + p[9] * X[:, 6])
    j[:, 5] = X[:, 0]
    j[:, 6] = X[:, 1]
    j[:, 7] = X[:, 2]
    return j

p_init = np.repeat(1, 10)
fit_aids_protein = optim.least_squares(fun_fit_aids, p_init, jac=jacobian_aids, args=(X, y, 'protein'), verbose=1)
fit_aids_fat = optim.least_squares(fun_fit_aids, p_init, jac=jacobian_aids, args=(X, y, 'fat'), verbose=1)
fit_aids_carb = optim.least_squares(fun_fit_aids, p_init, jac=jacobian_aids, args=(X, y, 'carb'), verbose=1)

# inference

n = X.shape[0]
k = p_init.size
sighat2 = (np.matrix(fit_aids.fun) * np.matrix(fit_aids.fun).T)[0, 0] / (n - k)
J = np.matrix(fit_aids.jac)
param_cov = sighat2 * np.linalg.inv(J.T * J)
param_se = np.sqrt(np.diag(param_cov))
param_tvals = np.divide(fit_aids.x, param_se)
t_crit = stats.t.ppf(.975, n - k)
param_pvals = 2 * (1 - stats.t.cdf(abs(param_tvals), n - k))
# own-price effect is only sig variable
# may be expected when analyzing at macronutrient level

# corrected stone index for OLS almost ideal demand system, giancarlo moschini (1995)

price_bidx = df.loc[p_df.index.tolist(), 'date'][df['date'] == 'Feb-13'].index.tolist()
price_base = np.empty((len(macro_price)))

for i in range(len(macro_price)):
    price_base[i] = macros.loc[price_bidx, macro_price[i]].mean()

cStone_pidx = ['Sprotein_idx', 'Sfat_idx', 'Scarb_idx']
for i in range(len(cStone_pidx)):
    p_df[cStone_pidx[i]] = macros.loc[p_df.index.tolist(), macro_props[i]] * (p_df[p_df.columns[i]] -
                                                                           np.log(price_base[i]))

X_cStoneidx = np.concatenate([np.ones(len(p_df)).reshape(len(p_df), 1),
                              np.array(p_df[['lnprotein_p', 'lnfat_p', 'lncarb_p']]),
                              np.array(df.loc[p_df.index.tolist(), 'total_fd_exp']).reshape(len(p_df), 1),
                              np.array(p_df[['Sprotein_idx', 'Sfat_idx', 'Scarb_idx']])],
                             axis=1)

def ols_est(X, y, indicate):
    y = np.asmatrix(y).T
    X = np.asmatrix(X)

    if indicate == 'params':
        return np.linalg.inv(X.T * X) * X.T * y

    elif indicate == 'resids':
        return y - X * np.linalg.inv(X.T * X) * X.T * y

    else:
        print('Indicate "parameters" or "residuals"')

ols_params_protein = ols_est(X_cStoneidx, y[:, 0], 'params')
ols_params_fat = ols_est(X_cStoneidx, y[:, 1], 'params')
ols_params_carb = ols_est(X_cStoneidx, y[:, 2], 'params')

# ols inference

ols_resids = ols_est(X_cStoneidx, y[:, 0], 'resids')
ols_sighat2 = ((ols_resids.T * ols_resids) / (X_cStoneidx.shape[0] - X_cStoneidx.shape[1]))[0, 0]
ols_paramcov = ols_sighat2 * np.linalg.inv(np.asmatrix(X_cStoneidx).T * np.asmatrix(X_cStoneidx))
ols_paramse = np.sqrt(np.diag(ols_paramcov))
ols_tvals = np.divide(np.array(ols_params).reshape(1, X_cStoneidx.shape[1]), ols_paramse)
ols_pvals = 2 * (1 - stats.t.cdf(abs(ols_tvals), X_cStoneidx.shape[0] - X_cStoneidx.shape[1]))
# all price effects, own and cross, are significant under OLS estimation

# nonlinear estimation using LaFrance (1990) incomplete demand specification

# converted str #VALUE! error in crop income to zero
df.loc[df.loc[p_df.index.tolist()][df['cropincome'] == '#VALUE!'].index.tolist(), 'cropincome'] = 0

p_df['total_inc'] = df.loc[p_df.index.tolist(),
                           [var for var in df.columns if 'ncome' in var]].astype('float64').sum(axis=1)

def m_sum(var):
    return macros[var] * macros[macro_price].sum(axis=1)

macros['protein_psum'] = m_sum('protein_p')
macros['fat_psum'] = m_sum('fat_p')
macros['carb_psum'] = m_sum('carb_p')

X_ids = np.concatenate([np.array(macros.loc[p_df.index.tolist(), macro_price]),
                        np.array(p_df['total_inc']).reshape(len(p_df), 1),
                        np.array(macros.loc[p_df.index.tolist(), ['protein_psum', 'fat_psum', 'carb_psum']])
                        ], axis=1)

y_ids = np.array(macros.loc[p_df.index.tolist(), ['protein_cons', 'fat_cons', 'carbs_cons']])

def fun_fit_ids(p, X, y, indicate):
    # alpha param 0-3
    # beta params 4-9
    # gamma 10-12

    if indicate == 'protein':
        dep = y[:, 0]

    elif indicate == 'fat':
        dep = y[:, 1]

    else:
        dep = y[:, 2]

    return (p[0] + p[4] * X[:, 0] + p[5] * X[:, 1] + p[6] * X[:, 2] +
            p[10] * (X[:, 3] - (p[1] * X[:, 0] + p[2] * X[:, 1] + p[3] * X[:, 2]) -
                     0.5 * (p[7] * X[:, 4] + p[8] * X[:, 5] + p[9] * X[:, 6])) +
            p[11] * (X[:, 3] - (p[1] * X[:, 0] + p[2] * X[:, 1] + p[3] * X[:, 2]) -
                     0.5 * (p[7] * X[:, 4] + p[8] * X[:, 5] + p[9] * X[:, 6])) +
            p[12] * (X[:, 3] - (p[1] * X[:, 0] + p[2] * X[:, 1] + p[3] * X[:, 2]) -
                     0.5 * (p[7] * X[:, 4] + p[8] * X[:, 5] + p[9] * X[:, 6]))
            ) - dep

def jacobian_ids(p, X, y, indicate):
    j = np.empty((X_ids.shape[0], p.size))
    j[:, 0] = 1
    j[:, 1] = -p[10] * X[:, 0] - p[11] * X[:, 0] - p[12] * X[:, 0]
    j[:, 2] = -p[10] * X[:, 1] - p[11] * X[:, 1] - p[12] * X[:, 1]
    j[:, 3] = -p[10] * X[:, 2] - p[11] * X[:, 2] - p[12] * X[:, 2]
    j[:, 4] = X[:, 0]
    j[:, 5] = X[:, 1]
    j[:, 6] = X[:, 2]
    j[:, 7] = -(p[10] + p[11] + p[12]) * 0.5 * X[:, 4]
    j[:, 8] = -(p[10] + p[11] + p[12]) * 0.5 * X[:, 5]
    j[:, 9] = -(p[10] + p[11] + p[12]) * 0.5 * X[:, 6]
    j[:, 10] = (X[:, 3] - (p[1] * X[:, 0] + p[2] * X[:, 1] + p[3] * X[:, 2]) -
                0.5 * (p[7] * X[:, 4] + p[8] * X[:, 5] + p[9] * X[:, 6]))
    j[:, 11] = (X[:, 3] - (p[1] * X[:, 0] + p[2] * X[:, 1] + p[3] * X[:, 2]) -
                0.5 * (p[7] * X[:, 4] + p[8] * X[:, 5] + p[9] * X[:, 6]))
    j[:, 12] = (X[:, 3] - (p[1] * X[:, 0] + p[2] * X[:, 1] + p[3] * X[:, 2]) -
                0.5 * (p[7] * X[:, 4] + p[8] * X[:, 5] + p[9] * X[:, 6]))
    return j

p_init = np.repeat(1, 13)
fit_ids_protein = optim.least_squares(fun_fit_ids, p_init, jac=jacobian_ids, args=(X_ids, y_ids, 'protein'), verbose=1)
fit_ids_fat = optim.least_squares(fun_fit_ids, p_init, jac=jacobian_ids, args=(X_ids, y_ids, 'fat'), verbose=1)
fit_ids_carb = optim.least_squares(fun_fit_ids, p_init, jac=jacobian_ids, args=(X_ids, y_ids, 'carb'), verbose=1)

# TODO ids parameter inference




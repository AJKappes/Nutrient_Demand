### Western Kenya Nutrient Demand Analysis ###

import numpy as np
import pandas as pd
import glob
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.figure_factory as ff
from scipy import stats
import scipy.optimize as optim


###################### Data management ##################################
#########################################################################

def files(sub_dir):
    return glob.glob('/home/ajkappes/' + sub_dir + '*.csv')

nutrient_dfs = np.array(files('/research/africa/Nutrient_Demand/'))
health_dfs = np.array(files('/research/africa/Livestock_Human_Health/data/'))
print(nutrient_dfs, '\n', health_dfs)

df_consump = pd.read_csv(nutrient_dfs[0])
df_nutrient_props = pd.read_csv(nutrient_dfs[4])
df_land_crop = pd.read_csv(nutrient_dfs[6])
df_hh_demographs = pd.read_csv(nutrient_dfs[5])
df_livinc = pd.read_csv(nutrient_dfs[3])
df_hh_asset = pd.read_csv(nutrient_dfs[2])
#df_healthcare = pd.read_csv(health_dfs[0])
#df_livestock_gen_health = pd.read_csv(health_dfs[1])
#df_livestock_health = pd.read_csv(health_dfs[2])

# Need to match HH_id by month due to repeating/missing id values across months
# then concatenate all months Feb 13 - Jul 16

m_y = pd.DataFrame(df_consump['date'].unique().tolist()).rename(columns={0: 'date'})

d = {}
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

        # df_healthcare[df_healthcare['date'] == key].drop_duplicates(['HousehldID'], keep='last'),
        # how='inner', on='HousehldID').merge()
        #
        # df_livestock_gen_health[df_livestock_gen_health['date'] == key].drop_duplicates(['HousehldID'], keep='last'),
        # how='inner', on='HousehldID').merge(
        #
        # df_livestock_health[df_livestock_health['date'] == key].drop_duplicates(['HousehldID'], keep='last'),
        # how='inner', on='HousehldID').dropna()


df_list = [d[m_y.loc[0, 'date']]]
i = 1
while i < len(d):
    df_list.append(d[m_y.loc[i, 'date']])
    i += 1
    if i > len(d):
        break

df = pd.concat(df_list, axis=0).reset_index().drop(columns='index')  # Ending with 15323 observations

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
micronutrients = ['iron', 'zinc', 'vit_a']
nutrients = macronutrients + micronutrients

goat_sheep_milk = pd.DataFrame(df_nutrient_props[df_nutrient_props['food_source'].isin(['milk_sheep', 'milk_goat'])]
                               [nutrients].mean()).T

pulses = pd.DataFrame(df_nutrient_props[df_nutrient_props['food_source'].isin(['peas', 'navy', 'pinto',
                                                                               'black', 'kidney', 'lentils'])]
                      [nutrients].mean()).T

viungo = pd.DataFrame(df_nutrient_props[df_nutrient_props['food_source'].isin(['onions', 'tomatoes', 'carrots',
                                                                               'green_peppers'])]
                      [nutrients].mean()).T

greens = pd.DataFrame(df_nutrient_props[df_nutrient_props['food_source'].isin(['spinach', 'cabbage'])]
                      [nutrients].mean()).T

nutrient_comps = pd.concat([df_nutrient_props[df_nutrient_props['food_source'] == 'milk_cow'][nutrients],
                            goat_sheep_milk,
                            df_nutrient_props[df_nutrient_props['food_source'] == 'eggs'][nutrients],
                            df_nutrient_props[df_nutrient_props['food_source'] == 'beef'][nutrients],
                            df_nutrient_props[df_nutrient_props['food_source'] == 'goat'][nutrients],
                            df_nutrient_props[df_nutrient_props['food_source'] == 'other_poultry'][nutrients],
                            df_nutrient_props[df_nutrient_props['food_source'] == 'fish'][nutrients],
                            df_nutrient_props[df_nutrient_props['food_source'] == 'maize'][nutrients],
                            df_nutrient_props[df_nutrient_props['food_source'] == 'cassava'][nutrients],
                            df_nutrient_props[df_nutrient_props['food_source'] == 'sorghum'][nutrients],
                            df_nutrient_props[df_nutrient_props['food_source'] == 'banana'][nutrients],
                            pulses, viungo, greens,
                            df_nutrient_props[df_nutrient_props['food_source'] == 'potato'][nutrients],
                            df_nutrient_props[df_nutrient_props['food_source'] == 'oil'][nutrients]],
                           axis=0).reset_index().drop(columns='index')

# Macronutrient proportions are values per 100g of food item consumed
# conversions: 1000g[water] per L
#              1000g per Kg
#
# measured in L: cow and goat/sheep milk
# measured in Kg: all meat and crop items
# eggs measured in # consumed: medium egg = 44g => scale 100g egg nutrient values by 0.44
#
# Micronutrient: iron and zinc in mg, vitamin A in micrograms
#                conversions 1:.001 and 1:1e-6

conversion = np.array([10, 10, 0.44, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10])
consumption = np.array(consumption)
nutrient_comps = np.array(nutrient_comps)

nutr_cons = pd.DataFrame({'protein_cons': np.zeros(len(consumption)),
                          'fat_cons': np.zeros(len(consumption)),
                          'carb_cons': np.zeros(len(consumption)),
                          'iron_cons': np.zeros(len(consumption)),
                          'zinc_cons': np.zeros(len(consumption)),
                          'vit_a_cons': np.zeros(len(consumption))})

for i in range(consumption.shape[0]):
    nutr_cons.loc[i, 'protein_cons'] = np.dot(np.multiply(nutrient_comps[:, 0], consumption[i]), conversion)
    nutr_cons.loc[i, 'fat_cons'] = np.dot(np.multiply(nutrient_comps[:, 1], consumption[i]), conversion)
    nutr_cons.loc[i, 'carb_cons'] = np.dot(np.multiply(nutrient_comps[:, 2], consumption[i]), conversion)
    nutr_cons.loc[i, 'iron_cons'] = np.dot(np.multiply(nutrient_comps[:, 3], consumption[i]), conversion)
    nutr_cons.loc[i, 'zinc_cons'] = np.dot(np.multiply(nutrient_comps[:, 4], consumption[i]), conversion)
    nutr_cons.loc[i, 'vit_a_cons'] = np.dot(np.multiply(nutrient_comps[:, 5], consumption[i]), conversion)

# consumption values listed for micronutrients are in milligrams and micrograms (others in grams)

# shadow price construction
food_exp_list = [var for var in df.columns if 'Cost' in var][:-5] # removing non-food exps
df['total_fd_exp'] = df[food_exp_list].sum(axis=1) # total cost of food across all foods consumed

# each nutrient's proportion of total food expense
macro_props = ['protein_prop', 'fat_prop', 'carb_prop']
for i in range(len(macro_props)):
    nutr_cons[macro_props[i]] = nutr_cons[nutr_cons.columns[i]] / nutr_cons[['protein_cons', 'fat_cons',
                                                                             'carb_cons']].sum(axis=1)

# think about micro proportions - iron and zinc units differ from vit_a units
micro_props = ['iron_prop', 'zinc_prop', 'vit_a_prop']
for i in range(len(micro_props)):
    nutr_cons[micro_props[i]] = nutr_cons[nutr_cons.columns[i + 3]] / nutr_cons[['iron_cons', 'zinc_cons',
                                                                                 'vit_a_cons']].sum(axis=1)

# nutrient shadow price
nutrient_price = []
for i in nutrients:
    nutrient_price.append(i + '_p')

for i in range(len(nutrient_price)):
    nutr_cons[nutrient_price[i]] = df['total_fd_exp'] / nutr_cons[nutr_cons.columns[i]]

nutr_cons = nutr_cons.replace([np.inf, -np.inf], np.nan).dropna()
macro_price = nutrient_price[0:3]
micro_price = nutrient_price[3:]

macros_l = []
for i in macronutrients:
    macros_l.append(i + '_cons')
    macros_l.append(i + '_p')
    macros_l.append(i + '_prop')
    macros = nutr_cons[macros_l]

micros_l = []
for i in micronutrients:
    micros_l.append(i + '_cons')
    micros_l.append(i + '_p')
    micros = nutr_cons[micros_l]

#macronutrient consumption total cost
macro_cost = ['protein_cost', 'fat_cost', 'carb_cost']
for i in range(len(macro_cost)):
    macros[macro_cost[i]] = macros[macro_props[i]] * df['total_fd_exp']

# shadow prices df
p_df = nutr_cons.loc[~(nutr_cons[nutrient_price] == 0).all(axis=1), nutrient_price]

lnp_list = ['lnprotein_p', 'lnfat_p', 'lncarb_p', 'lniron_p', 'lnzinc_p', 'lnvit_a_p']
for i in range(len(lnp_list)):
    p_df[lnp_list[i]] = np.log(p_df[p_df.columns[i]])

p_df = pd.concat([df.loc[p_df.index.tolist(), ['date', 'HousehldID']],
                  p_df,
                  nutr_cons.loc[p_df.index.tolist(), macro_props + micro_props],
                  nutr_cons.loc[p_df.index.tolist(), [j for j in nutr_cons.columns if '_cons' in j]],
                  df.loc[p_df.index.tolist(), 'total_fd_exp']], axis=1)

macro_budg_share = ['protein_bs', 'fat_bs', 'carb_bs']
for i in range(len(macro_budg_share)):
    p_df[macro_budg_share[i]] = p_df[nutrient_price[i]] * p_df[macronutrients[i] + '_cons'] * p_df[macro_props[i]]

#p_df.to_csv('/home/ajkappes/research/africa/Nutrient_Demand/nutrient_shadow_pricing.csv')

macro_lnp = p_df.columns[0:3].tolist()
def d_sum(var, df_col):
    return p_df[var] * p_df[df_col].sum(axis=1)

p_df['lnprotein_sum'] = d_sum('lnprotein_p', macro_lnp)
p_df['lnfat_sum'] = d_sum('lnfat_p', macro_lnp)
p_df['lncarb_sum'] = d_sum('lncarb_p', macro_lnp)

# household demographics df
dem_df = pd.DataFrame({'elec': df.loc[p_df.index.tolist(), 'Electricity'],
                       'bore_dam': 0,
                       'tap': 0,
                       'no_edu': 0,
                       'prim_sch': 0,
                       'sec_sch': 0,
                       'total_hh_mem': df.loc[p_df.index.tolist(), 'TotalHHMembers'],
                       'hh_mem_sq': df.loc[p_df.index.tolist(), 'TotalHHMembers'] ** 2,
                       'crop_acres': df.loc[p_df.index.tolist(), 'CropAcres']})

dem_df.loc[df.loc[p_df.index.tolist()][df['WaterSource'].isin([1, 2, 4])].index.tolist(), 'bore_dam'] = 1
dem_df.loc[df.loc[p_df.index.tolist()][df['WaterSource'] == 5].index.tolist(), 'tap'] = 1
dem_df.loc[df.loc[p_df.index.tolist()][df['maxedu'] == 1].index.tolist(), 'no_edu'] = 1
dem_df.loc[df.loc[p_df.index.tolist()][df['maxedu'] == 2].index.tolist(), 'prim_sch'] = 1
dem_df.loc[df.loc[p_df.index.tolist()][df['maxedu'] == 3].index.tolist(), 'sec_sch'] = 1

# converted str #VALUE! error in crop income to zero
df.loc[df.loc[p_df.index.tolist()][df['cropincome'] == '#VALUE!'].index.tolist(), 'cropincome'] = 0

dem_df['total_inc'] = df.loc[p_df.index.tolist(),
                             [var for var in df.columns if 'ncome' in var]].astype('float64').sum(axis=1)


###### descriptive stats ######

# variable stats
var_stats_cols = ['protein_prop', 'fat_prop', 'carb_prop',
                  'protein_p', 'fat_p', 'carb_p',
                  'total_hh_mem', 'crop_acres']

var_stats_df = pd.DataFrame(columns=var_stats_cols, index=['min', 'max', 'mean', 'sd'])

for var in var_stats_cols:
    if 'protein' in var or 'fat' in var or 'carb' in var:
        var_stats_df[var] = np.array([macros.loc[p_df.index.tolist(), var].min(),
                                      macros.loc[p_df.index.tolist(), var].max(),
                                      macros.loc[p_df.index.tolist(), var].mean(),
                                      macros.loc[p_df.index.tolist(), var].std()]).round(5)
    else:
        var_stats_df[var] = np.array([dem_df[var].min(),
                                      dem_df[var].max(),
                                      dem_df[var].mean(),
                                      dem_df[var].std()]).round(5)

# macronutrient monthly mean stats
nutr_idx = {}
for month_year in m_y['date']:
    nutr_idx[month_year] = pd.DataFrame()

for key in nutr_idx:
    nutr_idx[key] = df[df['date'] == key].index.tolist()

d_macros = {}
d_micros = {}
for month_year in m_y['date']:
    d_macros[month_year] = pd.DataFrame()
    d_micros[month_year] = pd.DataFrame()

for key in d_macros:
    d_macros[key] = macros.loc[nutr_idx[key]].mean()
    d_micros[key] = micros.loc[nutr_idx[key]].mean()

df_macro_statlist = [d_macros[m_y.loc[0, 'date']]]
df_micro_statlist = [d_micros[m_y.loc[0, 'date']]]
i = 1
while i < len(d_macros):
    df_macro_statlist.append(d_macros[m_y.loc[i, 'date']])
    df_micro_statlist.append(d_micros[m_y.loc[i, 'date']])
    i += 1
    if i > len(d_macros):
        break

df_macrostat = pd.DataFrame(pd.concat(df_macro_statlist, axis=1).T).round(3)
df_microstat = pd.DataFrame(pd.concat(df_micro_statlist, axis=1).T).round(3)
df_macrostat.index, df_microstat.index = m_y['date'], m_y['date']


# macronutrient means plot
x_dates = df_macrostat.index

trace_p = go.Scatter(
    x=x_dates,
    y=df_macrostat['protein_cons'],
    mode='lines',
    name='Protein'
)

trace_f = go.Scatter(
    x=x_dates,
    y=df_macrostat['fat_cons'],
    mode='lines',
    name='Fat'
)

trace_c = go.Scatter(
    x=x_dates,
    y=df_macrostat['carbs_cons'],
    mode='lines',
    name='Carbohydrates'
)

line_data = [trace_p, trace_f, trace_c]
layout = dict(yaxis=dict(title='Macronutrient Consumption in Grams'),
              font=dict(family='Liberation Serif'))

figure = dict(data=line_data, layout=layout)
plotly.offline.plot(figure, filename='Macronutrient_means_plot.html')
plotly.io.write_image(figure, '/home/ajkappes/Research/Africa/Nutrient_Demand/LaTeX/mean_macro_cons.pdf')

# nutrient mean shadow prices plot
traces_macro = []
traces_micro = []
names_macro = ['Protein', 'Fat', 'Carbohydrates']
names_micro = ['Iron', 'Zinc']
for i in range(len(macro_price)):
    traces_macro.append(go.Scatter(
        x=x_dates,
        y=df_macrostat[macro_price[i]],
        mode='lines',
        name=names_macro[i]
    ))
for i in range(len(micro_price) - 1):
    traces_micro.append(go.Scatter(
        x=x_dates,
        y=df_microstat[micro_price[i]],
        mode='lines',
        name=names_micro[i]
    ))

def make_figure(nutrient, units):
    if nutrient == 'Macronutrient':
        layout = dict(yaxis=dict(title=nutrient + 'Shadow Price (' + units + ')'),
                      font=dict(family='Liberation Serif'))
        return dict(data=traces_macro, layout=layout)
    else:
        layout = dict(yaxis=dict(title=nutrient + 'Shadow Price (' + units + ')'),
                       font=dict(family='Liberation Serif'))
        return dict(data=traces_micro, layout=layout)

plotly.offline.plot(make_figure('Macronutrient', 'Ksh/g'), filename='Macronutrient_shadowprice_means_plot.html')
plotly.offline.plot(make_figure('Micronutrient', 'Ksh/mg'), filename='Micronutrient_shadowprice_means_plot.html')

plotly.io.write_image(make_figure('Macronutrient', 'Ksh/g'),
                      '/home/ajkappes/Research/Africa/Nutrient_Demand/LaTeX/mean_macro_shadowprice.pdf')
plotly.io.write_image(make_figure('Micronutrient', 'Ksh/mg'),
                      '/home/ajkappes/Research/Africa/Nutrient_Demand/LaTeX/mean_micro_shadowprice.pdf')

####################### demand estimation ###############################
#########################################################################

###### almost ideal demand system, deaton and muellbauer (1980) ######

def ols_est(X, y, indicate):
    y = np.asmatrix(y).T
    X = np.asmatrix(X)

    if indicate == 'params':
        return np.linalg.inv(X.T * X) * X.T * y

    elif indicate == 'resids':
        return y - X * np.linalg.inv(X.T * X) * X.T * y

    elif indicate == 'predict':
        return X * np.linalg.inv(X.T * X) * X.T * y

    else:
        print('Indicate "parameters", "residuals", or "predict"')

### corrected stone index for OLS almost ideal demand system, giancarlo moschini (1995) ###

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
                              np.array(p_df[['Sprotein_idx', 'Sfat_idx', 'Scarb_idx']]),
                              np.array(dem_df[['total_hh_mem', 'hh_mem_sq', 'crop_acres']])],
                             axis=1)

### Tornqvist index, giancarlo moschini (1995) ###

bshare_base = np.empty((len(macro_price)))
for i in range(len(macro_price)):
    bshare_base[i] = macros.loc[price_bidx, macro_props[i]].mean()

Tornidx = ['Tprotein_idx', 'Tfat_idx', 'Tcarb_idx']
for i in range(len(Tornidx)):
    p_df[Tornidx[i]] = 0.5 * ((macros.loc[p_df.index.tolist(), macro_props[i]] + bshare_base[i]) *
                              (p_df[p_df.columns[i]] - np.log(price_base[i])))

X_Tornidx = np.concatenate([np.ones(len(p_df)).reshape(len(p_df), 1),
                            np.array(p_df[['lnprotein_p', 'lnfat_p', 'lncarb_p']]),
                            np.array(p_df[['Tprotein_idx', 'Tfat_idx', 'Tcarb_idx']]),
                            np.array(dem_df[['total_hh_mem', 'hh_mem_sq', 'crop_acres']])],
                           axis=1)

### ols estimation ###

y_aids = np.array(macros.loc[p_df.index.tolist(), ['protein_prop', 'fat_prop', 'carb_prop']])

ols_params_protein_S = np.array(ols_est(X_cStoneidx, y_aids[:, 0], 'params')).reshape(1, X_cStoneidx.shape[1])[0]
ols_params_fat_S = np.array(ols_est(X_cStoneidx, y_aids[:, 1], 'params')).reshape(1, X_cStoneidx.shape[1])[0]
ols_params_carb_S = np.array(ols_est(X_cStoneidx, y_aids[:, 2], 'params')).reshape(1, X_cStoneidx.shape[1])[0]

ols_params_protein_T = np.array(ols_est(X_Tornidx, y_aids[:, 0], 'params')).reshape(1, X_Tornidx.shape[1])[0]
ols_params_fat_T = np.array(ols_est(X_Tornidx, y_aids[:, 1], 'params')).reshape(1, X_Tornidx.shape[1])[0]
ols_params_carb_T = np.array(ols_est(X_Tornidx, y_aids[:, 2], 'params')).reshape(1, X_Tornidx.shape[1])[0]


### almost ideal demand system nonlinear estimation ###

X_aids = np.concatenate([np.array(p_df[['lnprotein_p', 'lnfat_p', 'lncarb_p']]),
                         np.array(df.loc[p_df.index.tolist(), 'total_fd_exp']).reshape(len(p_df), 1),
                         np.array(p_df[['lnprotein_sum', 'lnfat_sum', 'lncarb_sum']]),
                         np.array(dem_df[['total_hh_mem', 'hh_mem_sq', 'crop_acres']])],
                        axis=1)

# function specified by deaton and muellbauer (1980)

def fun_fit_aids(p, X, y, indicate):
    # alpha params 0-3
    # beta param 4, 10-12
    # gamma param 5-9
    if indicate == 'protein':
        return ((p[1] - p[4] * p[0]) + p[5] * X[:, 0] + p[6] * X[:, 1] + p[7] * X[:, 2] +
                p[4] * (X[:, 3] - (p[2] * X[:, 1] + p[3] * X[:, 2]) - 0.5 * (p[8] * X[:, 5] + p[9] * X[:, 6])) +
                p[10] * X[:, 7] + p[11] * X[:, 8] + p[12] * X[:, 9]
                ) - y[:, 0]

    elif indicate == 'fat':
        return ((p[1] - p[4] * p[0]) + p[5] * X[:, 0] + p[6] * X[:, 1] + p[7] * X[:, 2] +
                p[4] * (X[:, 3] - (p[2] * X[:, 0] + p[3] * X[:, 2]) - 0.5 * (p[8] * X[:, 4] + p[9] * X[:, 6])) +
                p[10] * X[:, 7] + p[11] * X[:, 8] + p[12] * X[:, 9]
                ) - y[:, 1]

    else:
        return ((p[1] - p[4] * p[0]) + p[5] * X[:, 0] + p[6] * X[:, 1] + p[7] * X[:, 2] +
                p[4] * (X[:, 3] - (p[2] * X[:, 0] + p[3] * X[:, 1]) - 0.5 * (p[8] * X[:, 4] + p[9] * X[:, 5])) +
                p[10] * X[:, 7] + p[11] * X[:, 8] + p[12] * X[:, 9]
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
    j[:, 10] = X[:, 7]
    j[:, 11] = X[:, 8]
    j[:, 12] = X[:, 9]
    return j

# TODO add hh dem starting values
p_init_protein = np.array([1, 0.3, 1, 1, -0.0000003, -0.1297, 0.0219, 0.0987, 0.0107, 0.02, 0.00001, 0.0, -0.00029])
p_init_fat = np.array([1, 0.3, 1, 1, -0.0000017, 0.0232, -0.0985, 0.1313, 0.0271, -0.0672, -0.00044, 0.00003, 0.00031])
p_init_carb = np.array([0, 0.3, 0, 0, 0.000002, 0.1065, 0.0766, -0.23, 0.0178, 0.0664, 0.00043, -0.00003, -0.00002])

fit_aids_protein = optim.least_squares(fun_fit_aids, p_init_protein, jac=jacobian_aids, args=(X_aids, y_aids, 'protein'), verbose=1)
fit_aids_fat = optim.least_squares(fun_fit_aids, p_init_fat, jac=jacobian_aids, args=(X_aids, y_aids, 'fat'), verbose=1)
fit_aids_carb = optim.least_squares(fun_fit_aids, p_init_carb, jac=jacobian_aids, args=(X_aids, y_aids, 'carb'), verbose=1)


##### nonlinear estimation using LaFrance (1990) incomplete demand specification #####

# address endogeneity, deflate shadow prices and instrument with demographic variables
# deflate using first month mean
defl_p = ['defl_protein_p', 'defl_fat_p', 'defl_carb_p']
for i in range(len(defl_p)):
    macros[defl_p[i]] = macros[macro_price[i]] / price_base[i]

# macronutrient shadow price instrument construction
X_ins_proj = np.concatenate([np.ones(len(p_df)).reshape(len(p_df), 1),
                             np.array(dem_df[[var for var in dem_df.columns if 'total_inc' != var]])],
                            axis=1)

y_ins_proj = np.array(macros.loc[p_df.index.tolist(), defl_p])

ins_protein_p = ols_est(X_ins_proj, y_ins_proj[:, 0], 'predict')
ins_fat_p = ols_est(X_ins_proj, y_ins_proj[:, 1], 'predict')
ins_carb_p = ols_est(X_ins_proj, y_ins_proj[:, 2], 'predict')

ins_price_vals = [ins_protein_p, ins_fat_p, ins_carb_p]
ins_prices = ['ins_protein_p', 'ins_fat_p', 'ins_carb_p']
for i in range(len(ins_prices)):
    p_df[ins_prices[i]] = ins_price_vals[i]

p_df['lnins_protein_p'] = np.log(p_df['ins_protein_p'])
p_df['lnins_fat_p'] = np.log(p_df['ins_fat_p'])
p_df['lnins_carb_p'] = np.log(p_df['ins_carb_p'])

p_df['ins_protein_psum'] = d_sum('ins_protein_p', ins_prices)
p_df['ins_fat_psum'] = d_sum('ins_fat_p', ins_prices)
p_df['ins_carb_psum'] = d_sum('ins_carb_p', ins_prices)

# p_df = p_df[(p_df[ins_prices] >= 0).all(1)]
# dem_df = dem_df.loc[p_df.index.tolist()]

# unobervable commodity z consumption consistent estimators for instrumenting z in ids
X_z_ins = np.concatenate([np.ones(len(p_df)).reshape(len(p_df), 1),
                          np.array(dem_df[['total_hh_mem', 'crop_acres']]),
                          np.array(p_df[ins_prices]),
                          #np.array(p_df[['ins_protein_psum', 'ins_fat_psum', 'ins_carb_psum']])
                          ], axis=1)

protein_z_params = ols_est(X_z_ins, macros.loc[p_df.index.tolist(), 'protein_cons'], 'params')
fat_z_params = ols_est(X_z_ins, macros.loc[p_df.index.tolist(), 'fat_cons'], 'params')
carb_z_params = ols_est(X_z_ins, macros.loc[p_df.index.tolist(), 'carbs_cons'], 'params')

# ids construction

def opc_sum(var):
    return p_df[var] * dem_df[['total_hh_mem', 'crop_acres']].sum(axis=1)

dem_df['opc_dem_p'] = opc_sum('ins_protein_p')
dem_df['opc_dem_f'] = opc_sum('ins_fat_p')
dem_df['opc_dem_c'] = opc_sum('ins_carb_p')

X_ids = np.concatenate([np.array(p_df[ins_prices]),
                        np.array(dem_df['total_inc']).reshape(len(p_df), 1),
                        np.array(p_df[['ins_protein_psum', 'ins_fat_psum', 'ins_carb_psum']]),
                        np.array(dem_df[['total_hh_mem', 'crop_acres', 'elec', 'opc_dem_p', 'opc_dem_f', 'opc_dem_c']]),
                        ], axis=1)

y_ids = np.array(macros.loc[p_df.index.tolist(), ['protein_cons', 'fat_cons', 'carbs_cons']])

# gamma initialization value

gam_list = np.array([['ins_protein_p', 'opc_dem_p', 'ins_protein_psum'],
                     ['ins_fat_p', 'opc_dem_f', 'ins_fat_psum'],
                     ['ins_carb_p', 'opc_dem_c', 'ins_carb_psum']])

gam_vals = pd.DataFrame({'gam_protein': np.zeros(len(p_df)),
                         'gam_fat': np.zeros(len(p_df)),
                         'gam_carb': np.zeros(len(p_df))}, index=p_df.index)

for i in range(gam_list.shape[0]):
    gam_vals[gam_vals.columns[i]] = (dem_df['total_inc'] - protein_z_params[3, 0] * p_df[gam_list[i][0]] -
                                     protein_z_params[3, 0] * dem_df[gam_list[i][1]] -
                                     0.5 * protein_z_params[6, 0] * p_df[gam_list[i][2]])

gam_params = np.empty(gam_list.shape[0])

for i in range(gam_list.shape[0]):
    gam_params[i] = ols_est(np.concatenate([np.ones(len(p_df)).reshape(len(p_df), 1),
                                            np.array(gam_vals[gam_vals.columns[i]]).reshape(len(p_df), 1)], axis=1),
                            y_ids[:, 0], 'params')[1]

# ids specification

def fun_fit_ids(p, X, y, indicate):
    # alpha param 0-3
    # beta params 4-9
    # gamma params 10-12
    # delta params 13-15

    if indicate == 'protein':
        dep = y[:, 0]

    elif indicate == 'fat':
        dep = y[:, 1]

    else:
        dep = y[:, 2]

    return (p[0] + p[4] * X[:, 0] + p[5] * X[:, 1] + p[6] * X[:, 2] + p[13] * X[:, 7] + p[14] * X[:, 8] + p[15] * X[:, 9] +
            p[10] * (X[:, 3] - p[16] * X[:, 10] - p[1] * X[:, 0] - 0.5 * p[7] * X[:, 4]) +
            p[11] * (X[:, 3] - p[17] * X[:, 11] - p[2] * X[:, 1] - 0.5 * p[8] * X[:, 5]) +
            p[12] * (X[:, 3] - p[18] * X[:, 12] - p[3] * X[:, 2] - 0.5 * p[9] * X[:, 6])
            ) - dep

def jacobian_ids(p, X, y, indicate):
    j = np.empty((X_ids.shape[0], p.size))
    j[:, 0] = 1
    j[:, 1] = -(p[10] + p[11] + p[12]) * X[:, 0]
    j[:, 2] = -(p[10] + p[11] + p[12]) * X[:, 1]
    j[:, 3] = -(p[10] + p[11] + p[12]) * X[:, 2]
    j[:, 4] = X[:, 0]
    j[:, 5] = X[:, 1]
    j[:, 6] = X[:, 2]
    j[:, 7] = -(p[10] + p[11] + p[12]) * 0.5 * X[:, 4]
    j[:, 8] = -(p[10] + p[11] + p[12]) * 0.5 * X[:, 5]
    j[:, 9] = -(p[10] + p[11] + p[12]) * 0.5 * X[:, 6]
    j[:, 10] = X[:, 3] - p[15] * X[:, 9] - p[1] * X[:, 0] - 0.5 * p[7] * X[:, 4]
    j[:, 11] = X[:, 3] - p[16] * X[:, 10] - p[2] * X[:, 1] - 0.5 * p[8] * X[:, 5]
    j[:, 12] = X[:, 3] - p[17] * X[:, 11] - p[3] * X[:, 2] - 0.5 * p[9] * X[:, 6]
    j[:, 13] = X[:, 7]
    j[:, 14] = X[:, 8]
    j[:, 15] = X[:, 9]
    j[:, 16] = -(p[10] + p[11] + p[12]) * X[:, 10]
    j[:, 17] = -(p[10] + p[11] + p[12]) * X[:, 11]
    j[:, 18] = -(p[10] + p[11] + p[12]) * X[:, 12]
    return j

p_init_protein = np.array([5012.13, 1, 1, 1, -11183.58, 74.412, 7272.468,
                           1, 1, 1, 0.0057, 0.0079, 0.0032,
                           -51.958, 75.983, 1, 1, 1])

p_init_fat = np.array([903.633, 1430.592, 898.409, -2123.864, 1430.592, 898.409, -2123.864, 1430.592, 898.409, -2123.864,
                       1, 1, 1, 74.674, -61.839, 1430.592, 898.409, -2123.864])

p_init_carb = np.array([6359.043, 2776.27, 1228.477, -4900.755, 2776.27, 1228.477, -4900.755, 2776.27, 1228.477, -4900.755,
                        1, 1, 1, 652.833, 635.699, 2776.27, 1228.477, -4900.755])

fit_ids_protein = optim.least_squares(fun_fit_ids, p_init_protein, jac=jacobian_ids, args=(X_ids, y_ids, 'protein'), verbose=1)
fit_ids_fat = optim.least_squares(fun_fit_ids, p_init_fat, jac=jacobian_ids, args=(X_ids, y_ids, 'fat'), verbose=1)
fit_ids_carb = optim.least_squares(fun_fit_ids, p_init_carb, jac=jacobian_ids, args=(X_ids, y_ids, 'carb'), verbose=1)


##### parameter inference #####

class asym_param_inf:
    def __init__(self, fit, X):
        self.fit = fit
        self.X = np.matrix(X)

    def ols_param_cov(self, e):
        sighat2 = (e.T * e)[0, 0] / (self.X.shape[0] - self.X.shape[1])
        return sighat2 * np.linalg.inv(self.X.T * self.X)

    def nl_param_cov(self):
        sighat2 = (np.matrix(self.fit.fun) * np.matrix(self.fit.fun).T)[0, 0] / (self.X.shape[0] - len(self.fit.x))
        J = np.matrix(self.fit.jac)
        return sighat2 * np.linalg.inv(J.T * J)

    def param_se(self, cov):
        return np.sqrt(np.diag(cov))

    def param_t(self, se):
        return np.divide(self.fit.x, se)

    def param_p(self, t_vals):
        return 2 * (1 - stats.t.cdf(abs(t_vals), self.X.shape[0] - len(self.fit.x)))

class robust_param_inf:
    def __init__(self, X, e):
        self.X = np.matrix(X)
        self.e = e

    def nw_cov(self):
        n = self.X.shape[0]
        k = self.X.shape[1]

        S = np.matrix(np.zeros((k, k)))
        i = 0
        while i < n:
            S = S + n ** (-1) * (self.e[i, 0] ** 2 * self.X[i].reshape(k, 1) * self.X[i])
            i += 1
            if i > n:
                break

        W = np.matrix(np.zeros((k, k)))
        L = int(n ** (1 / 4))
        l = 0
        m = l + 1
        while m < n:
            W = W + n ** (-1) * ((L + 1 - l) / (L + 1) * self.e[m, 0] * self.e[m - 1, 0] *
                                 (self.X[m].reshape(k, 1) * self.X[m - 1] + self.X[m - 1].reshape(k, 1) * self.X[m]))
            m += 1
            if m > n:
                l += 1
                m = l + 1
            if l > L:
                break

        Q = S + W
        return n * np.linalg.inv(self.X.T * self.X) * Q * np.linalg.inv(self.X.T * self.X)

    def param_se(self, cov):
        return np.sqrt(np.diag(cov))

    def param_t(self, params, se):
        return np.divide(params, se)

    def param_p(self, t_vals, param_len):
        return 2 * (1 - stats.t.cdf(abs(t_vals), self.X.shape[0] - param_len))

# ols inference tables

ols_S_e = pd.DataFrame({'p_e': np.array(ols_est(X_cStoneidx, y_aids[:, 0], 'resids')[:, 0]).reshape(1, len(p_df))[0],
                        'f_e': np.array(ols_est(X_cStoneidx, y_aids[:, 1], 'resids')[:, 0]).reshape(1, len(p_df))[0],
                        'c_e': np.array(ols_est(X_cStoneidx, y_aids[:, 2], 'resids')[:, 0]).reshape(1, len(p_df))[0]},
                       index=p_df.index)

ols_T_e = pd.DataFrame({'p_e': np.array(ols_est(X_Tornidx, y_aids[:, 0], 'resids')[:, 0]).reshape(1, len(p_df))[0],
                        'f_e': np.array(ols_est(X_Tornidx, y_aids[:, 1], 'resids')[:, 0]).reshape(1, len(p_df))[0],
                        'c_e': np.array(ols_est(X_Tornidx, y_aids[:, 2], 'resids')[:, 0]).reshape(1, len(p_df))[0]},
                       index=p_df.index)

ols_param_inf_l = ['Stone_p', 'Stone_f', 'Stone_c', 'Torn_p', 'Torn_f', 'Torn_c']

S_inf_call = [robust_param_inf(X_cStoneidx, np.matrix(ols_S_e['p_e']).T),
              robust_param_inf(X_cStoneidx, np.matrix(ols_S_e['f_e']).T),
              robust_param_inf(X_cStoneidx, np.matrix(ols_S_e['c_e']).T)]

T_inf_call = [robust_param_inf(X_Tornidx, np.matrix(ols_S_e['p_e']).T),
              robust_param_inf(X_Tornidx, np.matrix(ols_S_e['f_e']).T),
              robust_param_inf(X_Tornidx, np.matrix(ols_S_e['c_e']).T)]

S_param_call = [ols_params_protein_S, ols_params_fat_S, ols_params_carb_S]

T_param_call = [ols_params_protein_T, ols_params_fat_T, ols_params_carb_T]

ols_param_se = {}
for key in ols_param_inf_l:
    ols_param_se[key] = pd.DataFrame()

ols_param_t = {}
for key in ols_param_inf_l:
    ols_param_t[key] = pd.DataFrame()

ols_param_pval = {}
for key in ols_param_inf_l:
    ols_param_pval[key] = pd.DataFrame()

for i in range(len(ols_S_e.columns)):
    ols_param_se[ols_param_inf_l[i]] = S_inf_call[i].param_se(S_inf_call[i].nw_cov())
    ols_param_se[ols_param_inf_l[i + 3]] = T_inf_call[i].param_se(T_inf_call[i].nw_cov())

for i in range(len(ols_S_e.columns)):
    ols_param_t[ols_param_inf_l[i]] = S_inf_call[i].param_t(S_param_call[i], ols_param_se[ols_param_inf_l[i]])
    ols_param_t[ols_param_inf_l[i + 3]] = T_inf_call[i].param_t(T_param_call[i], ols_param_se[ols_param_inf_l[i + 3]])

for i in range(len(ols_S_e.columns)):
    ols_param_pval[ols_param_inf_l[i]] = S_inf_call[i].param_p(ols_param_t[ols_param_inf_l[i]],
                                                               X_cStoneidx.shape[1])
    ols_param_pval[ols_param_inf_l[i + 3]] = T_inf_call[i].param_p(ols_param_t[ols_param_inf_l[i + 3]],
                                                                   X_Tornidx.shape[1])

def get_table(param_a, param_b, idx_a, idx_b):
    k = len(S_param_call[0])
    est_idx = ['intercept', 'protein_price', 'fat_price', 'carb_price', 'protein_idx', 'fat_idx',
               'carb_idx', 'total_hh_mem', 'total_hh_mem_sq', 'crop_acres']
    return pd.DataFrame({'stone_param': param_a,
                         'torn_param': param_b,
                         'stone_se': ols_param_se[idx_a].reshape(1, k)[0],
                         #'torn_se': ols_param_se[idx_b].reshape(1, k)[0],
                         'stone_t': ols_param_t[idx_a].reshape(1, k)[0],
                         #'torn_t': ols_param_t[idx_b].reshape(1, k)[0],
                         'stone_p': ols_param_pval[idx_a].reshape(1, k)[0]},
                         #'torn_p': ols_param_pval[idx_b].reshape(1, k)[0]},
                        index=est_idx)

protein_table = get_table(S_param_call[0], T_param_call[0], 'Stone_p', 'Torn_p').round(5)
fat_table = get_table(S_param_call[1], T_param_call[1], 'Stone_f', 'Torn_f').round(5)
carb_table = get_table(S_param_call[2], T_param_call[2], 'Stone_c', 'Torn_c').round(5)

# nonlinear inference table

nl_param_inf_l = ['protein' 'fat', 'carb']

nl_call = [robust_param_inf(fit_aids_protein.jac, np.matrix(fit_aids_protein.fun).T),
           robust_param_inf(fit_aids_fat.jac, np.matrix(fit_aids_fat.fun).T),
           robust_param_inf(fit_aids_carb.jac, np.matrix(fit_aids_carb.fun).T)]

nl_param_se = {}
for key in nl_param_inf_l:
    nl_param_se[key] = pd.DataFrame()

for i in range(len(nl_param_inf_l)):
    nl_param_se[nl_param_inf_l[i]] = nl_call[i].param_se(nl_call[i].nw_cov())

# nl standard error calculation error - parameter variance results are < 0
# TODO look further into optimal jacobian calculations and figure out variance problem

##### demand marginal price effects and elasticity ranges for OLS aids estimates #####

# marginal price effects
mean_prices = np.array(macros.loc[p_df.index.tolist(), macro_price].mean())
y_means = np.array(macros.loc[p_df.index.tolist(), macro_props].mean())

def p_effects(call, param_loc, mean_loc, y):
    return (call[param_loc] + call[param_loc + 3] * y) * 1 / mean_prices[mean_loc]

own_effects = np.empty(len(macro_price))
protein_c_effects = np.empty(len(macro_price) - 1)
fat_c_effects = np.empty(len(macro_price) - 1)
carb_c_effects = np.empty(len(macro_price) - 1)

for i in range(len(own_effects)):
    own_effects[i] = p_effects(S_param_call[i], i + 1, i, y_means[i])

for i in range(len(own_effects) - 1):
    protein_c_effects[i] = p_effects(S_param_call[0], i + 2, i + 1, y_means[0])
    carb_c_effects[i] = p_effects(S_param_call[2], i + 1, i, y_means[2])

    if i == 0:
        fat_c_effects[i] = p_effects(S_param_call[1], i + 1, i, y_means[1])
    else:
        fat_c_effects[i] = p_effects(S_param_call[1], i + 2, i + 1, y_means[1])

marg_effects = np.diag(own_effects)
marg_effects[0, 1:3] = protein_c_effects
marg_effects[1, [[0, 2]]] = fat_c_effects
marg_effects[2, 0:2] = carb_c_effects
marg_effects = pd.DataFrame(marg_effects, columns=macronutrients, index=macronutrients).round(4)

# elasticity distributions
def demand_elas(p, loc, y):
    return (p[loc] / y) + p[loc + 3]

ed_min = np.empty(len(macro_price))
ed_max = np.empty(len(macro_price))
ed_mean = np.empty(len(macro_price))
ed_sd = np.empty(len(macro_price))
for i in range(len(macro_price)):
    ed_min[i] = demand_elas(S_param_call[i], i + 1, y_aids[:, i]).min()
    ed_max[i] = demand_elas(S_param_call[i], i + 1, y_aids[:, i]).max()
    ed_mean[i] = demand_elas(S_param_call[i], i + 1, y_aids[:, i]).mean()
    ed_sd[i] = demand_elas(S_param_call[i], i + 1, y_aids[:, i]).std()


aggr_ed_summary = pd.DataFrame(np.stack([ed_min, ed_max, ed_mean, ed_sd]),
                               columns=['Protein', 'Fat', 'Carbohydrate'],
                               index=['min', 'max', 'mean', 'std_dev']).round(3)

# graphical analysis of elasticities over time
d_ed_idx = {}
for month_year in m_y['date']:
    d_ed_idx[month_year] = df[df['date'] == month_year].index

d_ed_p = np.empty(len(m_y))
d_ed_f = np.empty(len(m_y))
d_ed_c = np.empty(len(m_y))
for i in range(len(m_y)):
    d_ed_p[i] = macros.loc[d_ed_idx[m_y.loc[i, 'date']].intersection(p_df.index), macro_props[0]].mean()
    d_ed_f[i] = macros.loc[d_ed_idx[m_y.loc[i, 'date']].intersection(p_df.index), macro_props[1]].mean()
    d_ed_c[i] = macros.loc[d_ed_idx[m_y.loc[i, 'date']].intersection(p_df.index), macro_props[2]].mean()

ed_p = demand_elas(S_param_call[0], 1, d_ed_p)
ed_f = demand_elas(S_param_call[1], 2, d_ed_f)
ed_c = demand_elas(S_param_call[2], 3, d_ed_c)

trace_p_ed = go.Scatter(
    x=m_y['date'],
    y=ed_p,
    mode='lines',
    name='Protein elasticities'
)

trace_f_ed = go.Scatter(
    x=m_y['date'],
    y=ed_f,
    mode='lines',
    name='Fat elasticities'
)

trace_c_ed = go.Scatter(
    x=m_y['date'],
    y=ed_c,
    mode='lines',
    name='Carb elasticities'
)

mapping_ed = [trace_p_ed, trace_f_ed, trace_c_ed]
layout_ed = dict(yaxis=dict(title='Estimated Price Elasticity of Demand'),
                 font=dict(family='Liberation Serif'))

fig_ed = dict(data=mapping_ed, layout=layout_ed)
plotly.offline.plot(mapping_ed, filename='monthly_elasticities.html')
plotly.io.write_image(fig_ed, '/home/ajkappes/Research/Africa/Nutrient_Demand/LaTeX/monthly_elasticities.pdf')

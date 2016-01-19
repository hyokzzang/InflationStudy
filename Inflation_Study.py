# -----------------------------------------------------------------------------------
# Inflation Dynamics Analysis
#
# Author : Hyok Jung Kim
#
# Date and Version : 10th of January, 2016 (Ver 0.0)
#
# Purpose : Collect the Inflation Data, and analyze
#       (1) The inflation persistence of various countries
#       (2) The relationship between inflation persistence and the exchange rate persistence
#       (3) The correlation between consumption and the exchange rate
# -----------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import statsmodels.api as sm
import os
from metrics_tool import DatDef

# Turn off the unwanted warning message
pd.options.mode.chained_assignment = None

# The main directory of data
work_path = os.path.dirname(os.path.abspath(__file__))
cd_data = work_path + '/Data/'
cd_graph = work_path + '/Graphs/'

# Plot Title font settings
title_font = {'fontname':'Times New Roman', 'size':'16', 'color':'black', 'weight':'normal',
              'verticalalignment':'bottom'}

# Plot Axis font settings
axis_font = {'fontname':'Times New Roman', 'size':'16'}

# -----------------------------------------------------------------------------------
# Part 0-a: Data Read
# -----------------------------------------------------------------------------------

# Read the excel file of the inflation data
Data = pd.read_excel(cd_data+'Inflation_Countries.xlsx',
                     sheetname='raw',
                     header=0,
                     skiprows=None,
                     index_col=None,
                     parse_cols='A,D,E,F,G'
                     )

# Read the excel file containing country pool
CountryPool = pd.read_excel(cd_data+'Inflation_Countries.xlsx',
                            sheetname='country_dum',
                            header=0,
                            skiprows=None,
                            index_col=None
                            )

# Change the name of the columns
Data.rename(index=None, 
            columns={'LOCATION':'iso3','MEASURE':'measure','FREQUENCY':'freq','TIME':'time','Value':'CPI'}, 
            inplace=True)

# Merge Data with country pool
DataSort = pd.merge(Data, CountryPool, 
                    how='left', on=['iso3'])

# Leave only the data in the country pool, and with the quarterly frequency
DataSort = DataSort[(DataSort.country_dum==1)
                    & (DataSort.freq=='Q')
                    & (DataSort.measure=='AGRWTH')]

# -----------------------------------------------------------------------------------
# Part 0-b: Fetch sample period available for all countries
# -----------------------------------------------------------------------------------
# Groupby the data by iso3 code
Group_iso3 = DataSort.groupby(['iso3'])

# This is the data starting point for each country
minGroup_iso3 = Group_iso3.min()

# This becomes the starting data point available for all countries
sta_all = minGroup_iso3.max()
sta_all = sta_all.time

# This is the data endpoint for each country
maxGroup_iso3 = Group_iso3.max()

# This becomes the enddata point available for all countries
end_all = maxGroup_iso3.min()
end_all = end_all.time

# Encode time (but this filters the unique value of each time)
time_int, time_label = DataSort['time'].factorize(sort=True)
time_int = list(time_int)
time_label = list(time_label)

# Starting point, and endpoint as integer
sta_int = time_label.index(sta_all)
end_int = time_label.index(end_all)

# Convert time as an numeric variable
DataSort['time_encode'] = DataSort['time'].astype('category')
DataSort['time_numeric'] = DataSort['time_encode'].cat.codes

# Now filter the sample available for every country
DataSort = DataSort[(DataSort.time_numeric >= sta_int) & (DataSort.time_numeric <= end_int)]

# Drop unnecessary variables
DataSort.drop(['freq', 'country_dum', 'time_encode'], axis=1, inplace=True)

# As time series data
DataSort['t_index'] = pd.to_datetime(DataSort['time'])

# -----------------------------------------------------------------------------------
# Part 1: Graph Routine
# -----------------------------------------------------------------------------------

# Make an graph object
#   Note1: Colors are as follows
#           b: blue
#           g: green
#           r: red
#           c: cyan
#           m: magenta
#           y: yellow
#           k: black
#           w: white
#   Note2: Styles are,
#       - : solid line, o : circle dots, ^ : triangle, : : dots, :. : dots and line, -- : bar

# List of iso3 as the iterable list
iso3_list = CountryPool['iso3'].tolist()
N = len(iso3_list)

i = 1
N = len(iso3_list)
for x in iso3_list:
    
    # Extract CPI data for a given country
    CPItemp = DataSort['CPI'].loc[DataSort['iso3'] == x]

    # Extract the time series index for a given country
    t_temp = DataSort['t_index'].loc[DataSort['iso3'] == x]

    # Bug fix : depending on the pandas / matplotlib version,
    #           the arguments for plot should be strictly list, not the objects from pandas
    # temporary lists
    t_temp = t_temp.tolist()
    CPItemp = CPItemp.tolist()
    
    fig = plt.figure(i)
    ax = fig.add_subplot(111)
    plt.plot(t_temp, CPItemp, 'k-', linewidth = 1.0)
    
    # Make y label
    plt.ylabel(r'$\pi_t$', **axis_font)
    #plt.xlabel('')
    # Make a title
    plt.title(x, **title_font)
    # Adjust the tick size
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname('Times New Roman')
        label.set_fontsize(13)
    
    # Save the Graph
    plt.savefig(cd_graph+'graph_'+x+'.png')

    print(x, ':', i, "/", N)
    i = i+1
    plt.close(fig)

# -----------------------------------------------------------------------------------
# Part 2: Persistence
# -----------------------------------------------------------------------------------

# Lags
L = 3

# ---------------------------------------------------
# Part 2-a : Make lagged and differenced CPI data
# ---------------------------------------------------
# Define a panel object
panelObj = DatDef(DataSort, 'iso3', 'time_numeric')

# Give one lag for which persistence is estimated
DataSort = panelObj.lag(lags=1, targets='CPI')

# Define a panel object
panelObj = DatDef(DataSort, 'iso3', 'time_numeric')

# Make differences from 1 to 3 lags
DataSort = panelObj.diff(time=list(range(1,L+1)), targets='CPI')

# ---------------------------------------------------
# Part 2-b : Loop of calculating the inflation persistence
# ---------------------------------------------------

# Sort values by iso3 and time
DataSort.sort_values(by=['iso3', 'time_numeric'], axis=0, 
                     inplace=True, kind='mergesort', ascending=True)

# Initiate dictionary indicating persistence for each country
persistence_dict = {}

for x in iso3_list:
    # Extract the single country data
    y = DataSort.loc[DataSort['iso3']==x, 'CPI']
    
    # Pick X data
    X = DataSort.loc[DataSort['iso3']==x, ['L1_CPI', 'd1_CPI', 'd2_CPI', 'd3_CPI']]
    
    # Add constant
    X = sm.add_constant(X)
    
    # Restructure the data to drop NaN
    y = y.iloc[4:]
    X = X.iloc[4:,:]
    
    # OLS model
    persistence_model = sm.OLS(y, X)
    
    # Fit the OLS model
    results = persistence_model.fit()
    
    # Fit model again with HAC standard error
    results = results.get_robustcov_results(cov_type='HAC', maxlags = 1)
    
    # Gain the persistence
    persistence_dict[x] = results.params[1]
    

# DataFrame from dictionary
InflationPersistence = pd.DataFrame.from_dict(persistence_dict, orient='index')

# Make index as columns
InflationPersistence.reset_index(level=0, inplace=True)

# Rename index
InflationPersistence.rename(index=None, columns={'index':'iso3', 0:'InflationPersistence'}, inplace=True)

# ---------------------------------------------------
# Part 2-c : Plot inflation persistence with inflation level
# ---------------------------------------------------

# Groupby iso3
DataSortGroupiso3 = DataSort.groupby('iso3')

# Make mean by iso3
InflationMean = DataSortGroupiso3.mean().loc[:, 'CPI']

# Make index as columns
InflationMean = InflationMean.reset_index(level=0)

# Merge with persistence data
InflationMean = InflationMean.merge(InflationPersistence, on='iso3', how='left', copy=True)

corr = np.corrcoef(InflationMean['CPI'], InflationMean['InflationPersistence'])
corr = corr[0,1]
corr = np.around(corr, decimals=3)

fig = plt.figure(0)
ax = fig.add_subplot(111)
plt.scatter(InflationMean['CPI'], InflationMean['InflationPersistence'])
#list1 = InflationMean['CPI'].tolist()
#list2 = InflationMean['persistence'].tolist()
for label, x_val, y_val in zip(iso3_list, InflationMean['CPI'], InflationMean['InflationPersistence']):
    plt.annotate(label, xy=(x_val, y_val), xytext=(-20,5), textcoords='offset points',family='Times New Roman', size=10)

plt.text(x=0.6, y=0.1, s='Correlation Coefficient = ' + str(corr),
         family='Times New Roman', size=12, transform=ax.transAxes)

plt.text(x=0.6, y=0.05, s=r'$N$ = ' + str(N),
         family='Times New Roman', size=12, transform=ax.transAxes)

plt.xlabel(r'$\bar{\pi}_t$', **axis_font)
plt.ylabel('Inflation Persistence', **axis_font)
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontname('Times New Roman')
    label.set_fontsize(13)
plt.savefig(cd_graph+'graph_pit_persistence.png', dpi=300)
plt.close(fig)

# ---------------------------------------------------
# Part 2-d : Read Real Effective Exchange Rate Data, and Nominal Exchange Rate Data, Nominal Exchange Rate
#       Using the OECD Data
# ---------------------------------------------------

DataOECD = pd.read_excel(cd_data+'OECD_Data.xlsx',
                         sheetname='data',
                         header=0)

# Extract year and quarter, and fit to pandas time series format
DataOECD['year'] = DataOECD['time'].str[-4:]
DataOECD['quarter'] = DataOECD['time'].str[0:2]
DataOECD['time'] = DataOECD['year'].str.cat(DataOECD['quarter'], sep="-")

# As time series data
DataOECD['t_index'] = pd.to_datetime(DataOECD['time'])

# Drop unnecessary variables
DataOECD.drop(['year', 'quarter'], axis=1, inplace=True)

# Start and end point in the case of CPI data
CPI_sta = DataSort.loc[DataSort['iso3']=='AUS','t_index'].iloc[1]
CPI_end = DataSort.loc[DataSort['iso3']=='AUS','t_index'].iloc[-1]

# Factorize OECD data
DataOECD['t_index_num'], time_label = DataOECD['t_index'].factorize(sort=True)
DataOECD['t_index_num'] = pd.to_numeric(DataOECD['t_index_num'])

# As dictionary
timedict = pd.DataFrame(DataOECD['t_index_num'].tolist(), index=DataOECD['t_index'])
timedict = timedict[0].to_dict()

# Finally as time series
DataOECDSort = DataOECD.loc[(DataOECD['t_index_num']>timedict[CPI_sta]) & (DataOECD['t_index_num']<timedict[CPI_end]),:]

# Define a panel object
panelObj = DatDef(DataOECDSort, 'iso3', 't_index_num')

# Give one lag for which persistence is estimated
DataOECDSort = panelObj.lag(lags=1, targets='exrate_real_effective')

# Define a panel object
panelObj = DatDef(DataOECDSort, 'iso3', 't_index_num')

# Make differences from 1 to 3 lags
DataOECDSort = panelObj.diff(time=list(range(1,L+1)), targets='exrate_real_effective')

# drop Nan
DataOECDSort = DataOECDSort.loc[pd.notnull(DataOECDSort['exrate_real_effective']),:]

# Sort values by iso3 and time
DataOECDSort.sort_values(by=['iso3', 't_index_num'], axis=0, 
                         inplace=True, kind='mergesort', ascending=True)

iso3_list2 = list(DataOECDSort['iso3'].unique())

# Initiate dictionary indicating persistence for each country
exrate_persistence_dict = {}

for x in iso3_list2:
    # Extract the single country data
    y = DataOECDSort.loc[DataOECDSort['iso3']==x, 'exrate_real_effective']
    
    # Pick X data
    X = DataOECDSort.loc[DataOECDSort['iso3']==x, ['L1_exrate_real_effective', 'd1_exrate_real_effective', 'd2_exrate_real_effective', 'd3_exrate_real_effective']]
    
    # Add constant
    X = sm.add_constant(X)
    
    # Restructure the data to drop NaN
    y = y.iloc[4:]
    X = X.iloc[4:,:]
    
    # OLS model
    persistence_model = sm.OLS(y, X)
    
    # Fit the OLS model
    results = persistence_model.fit()
    
    # Fit model again with HAC standard error
    results = results.get_robustcov_results(cov_type='HAC', maxlags = 1)
    
    # Gain the persistence
    exrate_persistence_dict[x] = results.params[1]
    
# DataFrame from dictionary
ExratePersistence = pd.DataFrame.from_dict(exrate_persistence_dict, orient='index')

# Make index as columns
ExratePersistence.reset_index(level=0, inplace=True)

# Rename index
ExratePersistence.rename(index=None, columns={'index':'iso3', 0:'ExratePersistence'}, inplace=True)

PersistenceMerge = ExratePersistence.merge(InflationPersistence, on='iso3', how='inner')

iso3_list3 = list(PersistenceMerge['iso3'].unique())
N = len(iso3_list3)

corr = np.corrcoef(PersistenceMerge['InflationPersistence'], PersistenceMerge['ExratePersistence'])
corr = corr[0,1]
corr = np.around(corr, decimals=3)

fig = plt.figure(0)
ax = fig.add_subplot(111)
plt.scatter(PersistenceMerge['ExratePersistence'], PersistenceMerge['InflationPersistence'])
#list1 = InflationMean['CPI'].tolist()
#list2 = InflationMean['persistence'].tolist()
for label, x_val, y_val in zip(iso3_list3, PersistenceMerge['ExratePersistence'], PersistenceMerge['InflationPersistence']):
    plt.annotate(label, xy=(x_val, y_val), xytext=(-20,5), textcoords='offset points',family='Times New Roman', size=10)

plt.text(x=0.6, y=0.1, s='Correlation Coefficient = ' + str(corr),
         family='Times New Roman', size=12, transform=ax.transAxes)

plt.text(x=0.6, y=0.05, s=r'$N$ = ' + str(N),
         family='Times New Roman', size=12, transform=ax.transAxes)

plt.xlabel('Real Effective Exchange Rate Persistence (OECD)', **axis_font)
plt.ylabel('Inflation Persistence', **axis_font)
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontname('Times New Roman')
    label.set_fontsize(13)
plt.savefig(cd_graph+'graph_pit_exrate_persistence.png', dpi=300)
plt.close(fig)

##
# ---------------------------------------------------
# Part 2-e : Read Nominal Exchange Rate Data, Using the BIS Data
# ---------------------------------------------------

# Read Nominal Data
RawDataBISNominal = pd.read_excel(cd_data+'BIS(EffectiveRates).xlsx',
                         sheetname='Nominal',
                         header=0)

DataBISNominal = pd.melt(RawDataBISNominal, id_vars=['time'], var_name='iso3')

DataBISNominal['year'] = DataBISNominal['time'].str[-4:]
DataBISNominal['month'] = DataBISNominal['time'].str[:2]

MonthQuarterDict = {'01':'Q1', '02':'Q1', '03':'Q1','04':'Q2', '05':'Q2', '06':'Q2','07':'Q3', '08':'Q3', '09':'Q3','10':'Q4', '11':'Q4', '12':'Q4'}

MonthQuarter = pd.DataFrame.from_dict(MonthQuarterDict, orient='index')
MonthQuarter.reset_index(level=0, inplace=True)
MonthQuarter.rename(index=None, columns={'index':'month',0:'quarter'}, inplace=True)

#MonthQuarterDict = {'10':'04', '11':'04', '12':'04'}
DataBISNominal = DataBISNominal.merge(MonthQuarter, on='month', how='left', copy=True)

DataBISNominalgroup = DataBISNominal.groupby(['iso3', 'year', 'quarter'])

DataBISNominal = DataBISNominalgroup.mean()
DataBISNominal.reset_index(level=[0,1,2], inplace=True)
DataBISNominal['time'] = DataBISNominal['year'].str.cat(DataBISNominal['quarter'], sep="-")

# As time series data
DataBISNominal['t_index'] = pd.to_datetime(DataBISNominal['time'])

# Drop unnecessary variables
DataBISNominal.drop(['year', 'quarter'], axis=1, inplace=True)

# Factorize BIS data
DataBISNominal['t_index_num'], time_label = DataBISNominal['t_index'].factorize(sort=True)
DataBISNominal['t_index_num'] = pd.to_numeric(DataBISNominal['t_index_num'])

# As dictionary
timedict = pd.DataFrame(DataBISNominal['t_index_num'].tolist(), index=DataBISNominal['t_index'])
timedict = timedict[0].to_dict()

# Finally as time series
DataBISSort = DataBISNominal.loc[(DataBISNominal['t_index_num']>timedict[CPI_sta]) & (DataBISNominal['t_index_num']<timedict[CPI_end]),:]

# Define a panel object
panelObj = DatDef(DataBISSort, 'iso3', 't_index_num')

# Give one lag for which persistence is estimated
DataBISSort = panelObj.lag(lags=1, targets='value')

# Define a panel object
panelObj = DatDef(DataBISSort, 'iso3', 't_index_num')

# Make differences from 1 to 3 lags
DataBISSort = panelObj.diff(time=list(range(1,L+1)), targets='value')

# drop Nan
DataBISSort = DataBISSort.loc[pd.notnull(DataBISSort['value']),:]

# Sort values by iso3 and time
DataBISSort.sort_values(by=['iso3', 't_index_num'], axis=0, 
                         inplace=True, kind='mergesort', ascending=True)

iso3_list4 = list(DataBISSort['iso3'].unique())

# Initiate dictionary indicating persistence for each country
exrate_persistence_dict = {}

for x in iso3_list4:
    # Extract the single country data
    y = DataBISSort.loc[DataBISSort['iso3']==x, 'value']
    
    # Pick X data
    X = DataBISSort.loc[DataBISSort['iso3']==x, ['L1_value', 'd1_value', 'd2_value', 'd3_value']]
    
    # Add constant
    X = sm.add_constant(X)
    
    # Restructure the data to drop NaN
    y = y.iloc[4:]
    X = X.iloc[4:,:]
    
    # OLS model
    persistence_model = sm.OLS(y, X)
    
    # Fit the OLS model
    results = persistence_model.fit()
    
    # Fit model again with HAC standard error
    results = results.get_robustcov_results(cov_type='HAC', maxlags = 1)
    
    # Gain the persistence
    exrate_persistence_dict[x] = results.params[1]

# DataFrame from dictionary
ExratePersistence = pd.DataFrame.from_dict(exrate_persistence_dict, orient='index')

# Make index as columns
ExratePersistence.reset_index(level=0, inplace=True)

# Rename index
ExratePersistence.rename(index=None, columns={'index':'iso3', 0:'ExratePersistence'}, inplace=True)

PersistenceMerge = ExratePersistence.merge(InflationPersistence, on='iso3', how='inner')

iso3_list5 = list(PersistenceMerge['iso3'].unique())
N = len(iso3_list5)

corr = np.corrcoef(PersistenceMerge['InflationPersistence'], PersistenceMerge['ExratePersistence'])
corr = corr[0,1]
corr = np.around(corr, decimals=3)

fig = plt.figure(0)
ax = fig.add_subplot(111)
plt.scatter(PersistenceMerge['ExratePersistence'], PersistenceMerge['InflationPersistence'])
#list1 = InflationMean['CPI'].tolist()
#list2 = InflationMean['persistence'].tolist()
for label, x_val, y_val in zip(iso3_list5, PersistenceMerge['ExratePersistence'], PersistenceMerge['InflationPersistence']):
    plt.annotate(label, xy=(x_val, y_val), xytext=(-20,5), textcoords='offset points',family='Times New Roman', size=10)

plt.text(x=0.6, y=0.1, s='Correlation Coefficient = ' + str(corr),
         family='Times New Roman', size=12, transform=ax.transAxes)

plt.text(x=0.6, y=0.05, s=r'$N$ = ' + str(N),
         family='Times New Roman', size=12, transform=ax.transAxes)

plt.xlabel('Nominal Effective Exchange Rate Persistence (BIS)', **axis_font)
plt.ylabel('Inflation Persistence', **axis_font)
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontname('Times New Roman')
    label.set_fontsize(13)
plt.savefig(cd_graph+'graph_pit_exrate(nominal)_persistence.png', dpi=300)
plt.close(fig)

##
# ---------------------------------------------------
# Part 2-f : Read Real Effective Exchange Rate Data, using the BIS Data
# ---------------------------------------------------

# Read Real Data
RawDataBISReal = pd.read_excel(cd_data+'BIS(EffectiveRates).xlsx',
                         sheetname='Real',
                         header=0)

DataBISReal = pd.melt(RawDataBISReal, id_vars=['time'], var_name='iso3')

DataBISReal['year'] = DataBISReal['time'].str[-4:]
DataBISReal['month'] = DataBISReal['time'].str[:2]

MonthQuarterDict = {'01':'Q1', '02':'Q1', '03':'Q1','04':'Q2', '05':'Q2', '06':'Q2','07':'Q3', '08':'Q3', '09':'Q3','10':'Q4', '11':'Q4', '12':'Q4'}

MonthQuarter = pd.DataFrame.from_dict(MonthQuarterDict, orient='index')
MonthQuarter.reset_index(level=0, inplace=True)
MonthQuarter.rename(index=None, columns={'index':'month',0:'quarter'}, inplace=True)

#MonthQuarterDict = {'10':'04', '11':'04', '12':'04'}
DataBISReal = DataBISReal.merge(MonthQuarter, on='month', how='left', copy=True)

DataBISRealgroup = DataBISReal.groupby(['iso3', 'year', 'quarter'])

DataBISReal = DataBISRealgroup.mean()
DataBISReal.reset_index(level=[0,1,2], inplace=True)
DataBISReal['time'] = DataBISReal['year'].str.cat(DataBISReal['quarter'], sep="-")

# As time series data
DataBISReal['t_index'] = pd.to_datetime(DataBISReal['time'])

# Drop unnecessary variables
DataBISReal.drop(['year', 'quarter'], axis=1, inplace=True)

# Factorize BIS data
DataBISReal['t_index_num'], time_label = DataBISReal['t_index'].factorize(sort=True)
DataBISReal['t_index_num'] = pd.to_numeric(DataBISReal['t_index_num'])

# As dictionary
timedict = pd.DataFrame(DataBISReal['t_index_num'].tolist(), index=DataBISReal['t_index'])
timedict = timedict[0].to_dict()

# Finally as time series
DataBISSort = DataBISReal.loc[(DataBISReal['t_index_num']>timedict[CPI_sta]) & (DataBISReal['t_index_num']<timedict[CPI_end]),:]

# Define a panel object
panelObj = DatDef(DataBISSort, 'iso3', 't_index_num')

# Give one lag for which persistence is estimated
DataBISSort = panelObj.lag(lags=1, targets='value')

# Define a panel object
panelObj = DatDef(DataBISSort, 'iso3', 't_index_num')

# Make differences from 1 to 3 lags
DataBISSort = panelObj.diff(time=list(range(1,L+1)), targets='value')

# drop Nan
DataBISSort = DataBISSort.loc[pd.notnull(DataBISSort['value']),:]

# Sort values by iso3 and time
DataBISSort.sort_values(by=['iso3', 't_index_num'], axis=0, 
                         inplace=True, kind='mergesort', ascending=True)

iso3_list4 = list(DataBISSort['iso3'].unique())

# Initiate dictionary indicating persistence for each country
exrate_persistence_dict = {}

for x in iso3_list4:
    # Extract the single country data
    y = DataBISSort.loc[DataBISSort['iso3']==x, 'value']
    
    # Pick X data
    X = DataBISSort.loc[DataBISSort['iso3']==x, ['L1_value', 'd1_value', 'd2_value', 'd3_value']]
    
    # Add constant
    X = sm.add_constant(X)
    
    # Restructure the data to drop NaN
    y = y.iloc[4:]
    X = X.iloc[4:,:]
    
    # OLS model
    persistence_model = sm.OLS(y, X)
    
    # Fit the OLS model
    results = persistence_model.fit()
    
    # Fit model again with HAC standard error
    results = results.get_robustcov_results(cov_type='HAC', maxlags = 1)
    
    # Gain the persistence
    exrate_persistence_dict[x] = results.params[1]

# DataFrame from dictionary
ExratePersistence = pd.DataFrame.from_dict(exrate_persistence_dict, orient='index')

# Make index as columns
ExratePersistence.reset_index(level=0, inplace=True)

# Rename index
ExratePersistence.rename(index=None, columns={'index':'iso3', 0:'ExratePersistence'}, inplace=True)

PersistenceMerge = ExratePersistence.merge(InflationPersistence, on='iso3', how='inner')

iso3_list5 = list(PersistenceMerge['iso3'].unique())

N = len(iso3_list5)

corr = np.corrcoef(PersistenceMerge['InflationPersistence'], PersistenceMerge['ExratePersistence'])
corr = corr[0,1]
corr = np.around(corr, decimals=3)

fig = plt.figure(0)
ax = fig.add_subplot(111)
plt.scatter(PersistenceMerge['ExratePersistence'], PersistenceMerge['InflationPersistence'])
#list1 = InflationMean['CPI'].tolist()
#list2 = InflationMean['persistence'].tolist()
for label, x_val, y_val in zip(iso3_list5, PersistenceMerge['ExratePersistence'], PersistenceMerge['InflationPersistence']):
    plt.annotate(label, xy=(x_val, y_val), xytext=(-20,5), textcoords='offset points',family='Times New Roman', size=10)

plt.text(x=0.6, y=0.1, s='Correlation Coefficient = ' + str(corr),
         family='Times New Roman', size=12, transform=ax.transAxes)

plt.text(x=0.6, y=0.05, s=r'$N$ = ' + str(N),
         family='Times New Roman', size=12, transform=ax.transAxes)

plt.xlabel('Real Effective Exchange Rate Persistence (BIS)', **axis_font)
plt.ylabel('Inflation Persistence', **axis_font)
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontname('Times New Roman')
    label.set_fontsize(13)
plt.savefig(cd_graph+'graph_pit_exrate(real)_persistence.png', dpi=300)
plt.close(fig)


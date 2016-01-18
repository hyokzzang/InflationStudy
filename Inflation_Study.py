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
axis_font = {'fontname':'Times New Roman', 'size':'18'}

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
    plt.plot(t_temp, CPItemp, 'k-', linewidth = 1.0)
    
    # Make y label
    plt.ylabel(r'$\pi_t$', **axis_font)
    #plt.xlabel('')
    # Make a title
    plt.title(x, **title_font)
    # Adjust the tick size
    plt.tick_params(axis='both', which='major', labelsize=10)
    
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

# Define a panel object
panelObj = DatDef(DataSort, 'iso3', 'time_numeric')

# Give four lags
DataSort = panelObj.lag(lags=1, targets='CPI')

# Define a panel object
panelObj = DatDef(DataSort, 'iso3', 'time_numeric')

DataSort = panelObj.diff(time=list(range(1,L+1)), targets='CPI')

x = 'AUS'


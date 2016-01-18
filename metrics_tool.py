import pandas as pd

# Author : Hyok Jung Kim (Korea Institute of Industrial Economics and Trade)
# Version : 0.0.1
# Date : January 18th, 2016
# Description : This is a handy time series, and panel data management tool

class DatDef:
    
    # self = self
    # Input 1: DATA = Object as a pandas dataframe
    # Input 2: pidvar = Name of column of the pandas dataframe in the DATA object above
    #          pidvar can be a single string or list
    # Input 3: tvar = Name of column of the time series variable in the DATA object above
    #          it should have incrementing values
    def __init__(self, DATA, pidvar, tvar):
        self.DATA = DATA
        self.pidvar = pidvar
        self.tvar = tvar
    
    # --------------------------------------------------------------------------
    # lag : Function for making lags of variables
    # --------------------------------------------------------------------------
    def lag(self, lags, targets):
        # lags = lags can be given as list
        # targets = column list in the DATA object to be lagged
        
        # Test for list
        if not(type(self.pidvar) is list):
            self.pidvar = [self.pidvar]
        
        if not(type(self.tvar) is list):
            self.tvar = [self.tvar]
        
        if not(type(targets) is list):
            targets = [targets]
        
        if not(type(lags) is list):
            lags = [lags]
        
        EXTRACT_LIST = self.pidvar + self.tvar + targets
        
        # Extract pidvar, tvar, and targets from original data
        LAG_DAT = self.DATA.loc[:, EXTRACT_LIST].copy(deep=True)
        
        NEWDATA = self.DATA.copy(deep=True) 
        
        for l in lags:
            # Temporary data with lagged time index
            temp = LAG_DAT.copy(deep=True)
            temp.loc[:,self.tvar] = temp.loc[:,self.tvar] + l
            
            for x in targets:
                # New variable name such as L'l'_'targets'
                NewVarName = "L"+str(l)+"_"+str(x)
                
                # Apply new variable names
                temp.rename(index=None, columns={x:NewVarName}, inplace=True)
            
            # Finally merge into a new dataset
            NEWDATA = NEWDATA.merge(temp, how='left', on=self.pidvar+self.tvar)
            
        # Return the new dataset
        return NEWDATA
    
    # --------------------------------------------------------------------------
    # diff : Function for making differences of variables
    #       Note 1) 0 -> dx_t
    #            2) [0,1] -> dx_t, d_x_{t-1}
    # --------------------------------------------------------------------------    
    def diff(self, time, targets):
        
        # Test for list
        if not(type(self.pidvar) is list):
            self.pidvar = [self.pidvar]
        
        if not(type(self.tvar) is list):
            self.tvar = [self.tvar]
            
        if not(type(targets) is list):
            targets = [targets]
        
        if not(type(time) is list):
            time = [time]
        
        EXTRACT_LIST = self.pidvar + self.tvar + targets
        
        # Extract pidvar, tvar, and targets from original data
        DIFF_DAT = self.DATA.loc[:, EXTRACT_LIST].copy(deep=True)
        
        NEWDATA = self.DATA.copy(deep=True)
        
        for t in time:
            # Temporary data with lagged time index
            #temp = DIFF_DAT.copy(deep=True)
            
            lag_input = [t, t+1]
            
            tempObj = DatDef(DIFF_DAT, self.pidvar, self.tvar)
            L_tempObj = tempObj.lag(lag_input, targets)
            
            for x in targets:
                # New variable name such as L'l'_'targets'
                NewVarName = "d"+str(t)+"_"+str(x)
                
                L_tempObj[NewVarName] = L_tempObj["L"+str(t)+"_"+str(x)] - L_tempObj["L"+str(t+1)+"_"+str(x)]
                
                # Drop unwanted variables
                L_tempObj = L_tempObj.loc[:,self.pidvar+self.tvar+[NewVarName]]
            
                # Finally merge into a new dataset
                NEWDATA = NEWDATA.merge(L_tempObj, how='left', on=self.pidvar+self.tvar)
            
        # Return the new dataset
        return NEWDATA


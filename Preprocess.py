# -*- coding: utf-8 -*-

"""================================================
Import Packages
================================================"""

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog
import tkinter
from tkinter import Tk
from tkinter import simpledialog
from tkinter.filedialog import askopenfilename
from tkinter import colorchooser
from datetime import datetime
import os
#df=pd.read_csv('SampleData\\fund5_example_data_for_dt_V1.csv')
"""================================================
Import Sample file from Local
================================================"""

'''# chose your file'''
#root = tk.Tk()
#root.withdraw()
#file_path = filedialog.askopenfilename()
Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
filename = askopenfilename(initialdir = "/",title = "SELECT DATA FOR TRAINING",filetypes = (("jpeg files","*.jpg"),("all files","*.*"))) # show an "Open" dialog box and return the path to the selected file
#print(filename)


'''# read your file '''
df=pd.read_csv(filename)
ROOT = tk.Tk()
ROOT.withdraw()
# the input dialog
target_var = simpledialog.askstring(title="Test",
                                  prompt="ENTER TARGET VARIABLE:")

'''=======================================================================
Convert Target to 1/0:
    :as target variable y is yes and No, so 
     we need to convert it to numeric
======================================================================'''
#df['target']=df['y'].apply(lambda x: 1 if x=='yes' else 0)


''' try if Qdate is not available in dataframe''' 
try:   
    df.drop('QDate',axis=1,inplace=True) # droped y variable
except KeyError:
        pass

'''==============================================================================
3. # Get Numeric/non_neumeric Features
==============================================================================='''
numerics = ['int16', 'int32','int64', 'float16', 'float32', 'float64']
numerical_features = list(df.select_dtypes(include=numerics).columns)
Numeric_data = df[numerical_features]
Numeric_data.shape
numeric_feature=list(Numeric_data.columns)
#final_features=list(df[numeric_feature])


bool_ = ['bool']
bool_features = list(df.select_dtypes(include=bool_).columns)
bool_df=df[bool_features]



non_neumeric_df = df.drop(axis=1, labels=numeric_feature)
non_neumeric_df1=non_neumeric_df.drop(axis=1, labels=bool_features)
non_neumeric_feature=list(non_neumeric_df1.columns)



'''=======================================================================
Null Value Imputation
======================================================================'''

def missing_values_table(df):
        mis_val = df.isnull().sum()
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        return mis_val_table_ren_columns

mis_val_table_columns = missing_values_table(Numeric_data)
print(mis_val_table_columns)


'''=======================================================================
delete >= 70 % null values column Column
======================================================================'''
for c in Numeric_data:
    if 100*Numeric_data[c].isnull().sum()/len(Numeric_data) >= 70:
        Numeric_data.drop(c, axis=1, inplace=True)


mis_val_table_columns1 = missing_values_table(Numeric_data)

'''=======================================================================
:Mean Imputation
: 
======================================================================='''
Mis_Valu_befor_Imp=missing_values_table(Numeric_data) # check null value

mean_imp_numeric_data=Numeric_data.fillna(Numeric_data.mean())  
Null_val_after_imp=mean_imp_numeric_data[:].isnull().sum()
Mis_Valu_after_Mean_imp=missing_values_table(mean_imp_numeric_data)

'''
print('-'*50)
print("After mean imputation null vule column :::\n",Mis_Valu_after_Mean_imp)
print('-'*50)
'''

'''=======================================================================
 # Data transformation
    # Convert categorical values to numeric using label encoder
======================================================================='''

from sklearn import preprocessing
from collections import defaultdict
d = defaultdict(preprocessing.LabelEncoder)

'''# Encoding the categorical variable'''
fit = non_neumeric_df1.select_dtypes(include=['object']).fillna('NA').apply(lambda x: d[x.name].fit_transform(x))

'''#Convert the categorical columns based on encoding'''
try:
    for i in list(d.keys()):
        non_neumeric_df1[i] = d[i].transform(non_neumeric_df1[i].fillna('NA'))
except KeyError:
    pass
    
'''#Convert the categorical columns based on encoding'''

fit_bool = bool_df.select_dtypes(include=['bool']).fillna('NA').apply(lambda x: d[x.name].fit_transform(x))

"""    
'''=======================================================================
:Normalize float_Df_1_Not_Nul data
: already data has stadardize 
======================================================================='''  
Numeric_features_df = mean_imp_numeric_data[mean_imp_numeric_data.columns.difference([target_var])]  
def norm(df):
     result = df.copy()
     for feature_name in df.columns:
         max_value = df[feature_name].max()
         min_value = df[feature_name].min()
         result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
     return result
norm_numeric_df=norm(Numeric_features_df)
print('-'*50)
print("Normalised col::\n",list(scl_float_Df_1.columns.values))
print('-'*50)   
    
"""

'''=======================================================================
Concat imputed data-numeric and nonnumeric
    
======================================================================='''
final_df=pd.concat([non_neumeric_df1,fit_bool, mean_imp_numeric_data], axis=1)

''' drop zero values columns''' 
try:   
    final_df.drop(0,axis=1,inplace=True) # drop o values column
except KeyError:
        pass



''' ==================================================================
Correlation Matrix 
========================================================================'''

today = datetime.now()
mkdir=os.getcwd()+"//3_0_Correlation_files//"+today.strftime("%Y-%m-%d-%H%M%S")
if not os.path.exists(mkdir):
    os.makedirs(mkdir)
    
import seaborn as sns
import matplotlib.pyplot as plt
corr=final_df.corr()

sns.heatmap(corr,xticklabels=corr.columns,yticklabels=corr.columns)
corr.to_csv(mkdir +'\CorrelationTable.csv')
plt.savefig(mkdir +'\Correlation_Plot.png')

#------------------------------------
#------------------------------------
#------------------------------------
#------------------------------------


'''===================================================================
    FEATURE ENGINEERING:
        
        :Variable Selection using Python - Vote based approach

==================================================================='''
features = final_df[final_df.columns.difference([target_var])]
labels = final_df[[target_var]]
class_list=list(np.unique(df[[target_var]]))
#features = features.fillna(0)
total_feature=len(final_df.columns)

'''-------------------------------------------------------------------
1.WOE and IV
*******************Note: It is only applicable BINARY CLASSIFICATION
-------------------------------------------------------------------'''
# import packages
import pandas as pd
import numpy as np
import pandas.core.algorithms as algos
from pandas import Series
import scipy.stats.stats as stats
import re
import traceback
import string

max_bin = 20
force_bin = 3
# define a binning function

# define a binning function
def mono_bin(Y, X, n = max_bin):
    
    df1 = pd.DataFrame({"X": X, "Y": Y})
    justmiss = df1[['X','Y']][df1.X.isnull()]
    notmiss = df1[['X','Y']][df1.X.notnull()]
    r = 0
    while np.abs(r) < 1:
        try:
            d1 = pd.DataFrame({"X": notmiss.X, "Y": notmiss.Y, "Bucket": pd.qcut(notmiss.X, n)})
            d2 = d1.groupby('Bucket', as_index=True)
            r, p = stats.spearmanr(d2.mean().X, d2.mean().Y)
            n = n - 1 
        except Exception as e:
            n = n - 1

    if len(d2) == 1:
        n = force_bin         
        bins = algos.quantile(notmiss.X, np.linspace(0, 1, n))
        if len(np.unique(bins)) == 2:
            bins = np.insert(bins, 0, 1)
            bins[1] = bins[1]-(bins[1]/2)
        d1 = pd.DataFrame({"X": notmiss.X, "Y": notmiss.Y, "Bucket": pd.cut(notmiss.X, np.unique(bins),include_lowest=True)}) 
        d2 = d1.groupby('Bucket', as_index=True)
    
    d3 = pd.DataFrame({},index=[])
    d3["MIN_VALUE"] = d2.min().X
    d3["MAX_VALUE"] = d2.max().X
    d3["COUNT"] = d2.count().Y
    d3["EVENT"] = d2.sum().Y
    d3["NONEVENT"] = d2.count().Y - d2.sum().Y
    d3=d3.reset_index(drop=True)
    
    if len(justmiss.index) > 0:
        d4 = pd.DataFrame({'MIN_VALUE':np.nan},index=[0])
        d4["MAX_VALUE"] = np.nan
        d4["COUNT"] = justmiss.count().Y
        d4["EVENT"] = justmiss.sum().Y
        d4["NONEVENT"] = justmiss.count().Y - justmiss.sum().Y
        d3 = d3.append(d4,ignore_index=True)
    
    d3["EVENT_RATE"] = d3.EVENT/d3.COUNT
    d3["NON_EVENT_RATE"] = d3.NONEVENT/d3.COUNT
    d3["DIST_EVENT"] = d3.EVENT/d3.sum().EVENT
    d3["DIST_NON_EVENT"] = d3.NONEVENT/d3.sum().NONEVENT
    d3["WOE"] = np.log(d3.DIST_EVENT/d3.DIST_NON_EVENT)
    d3["IV"] = (d3.DIST_EVENT-d3.DIST_NON_EVENT)*np.log(d3.DIST_EVENT/d3.DIST_NON_EVENT)
    d3["VAR_NAME"] = "VAR"
    d3 = d3[['VAR_NAME','MIN_VALUE', 'MAX_VALUE', 'COUNT', 'EVENT', 'EVENT_RATE', 'NONEVENT', 'NON_EVENT_RATE', 'DIST_EVENT','DIST_NON_EVENT','WOE', 'IV']]       
    d3 = d3.replace([np.inf, -np.inf], 0)
    d3.IV = d3.IV.sum()
    
    return(d3)
def char_bin(Y, X):
        
    df1 = pd.DataFrame({"X": X, "Y": Y})
    justmiss = df1[['X','Y']][df1.X.isnull()]
    notmiss = df1[['X','Y']][df1.X.notnull()]    
    df2 = notmiss.groupby('X',as_index=True)
    
    d3 = pd.DataFrame({},index=[])
    d3["COUNT"] = df2.count().Y
    d3["MIN_VALUE"] = df2.sum().Y.index
    d3["MAX_VALUE"] = d3["MIN_VALUE"]
    d3["EVENT"] = df2.sum().Y
    d3["NONEVENT"] = df2.count().Y - df2.sum().Y
    
    if len(justmiss.index) > 0:
        d4 = pd.DataFrame({'MIN_VALUE':np.nan},index=[0])
        d4["MAX_VALUE"] = np.nan
        d4["COUNT"] = justmiss.count().Y
        d4["EVENT"] = justmiss.sum().Y
        d4["NONEVENT"] = justmiss.count().Y - justmiss.sum().Y
        d3 = d3.append(d4,ignore_index=True)
    
    d3["EVENT_RATE"] = d3.EVENT/d3.COUNT
    d3["NON_EVENT_RATE"] = d3.NONEVENT/d3.COUNT
    d3["DIST_EVENT"] = d3.EVENT/d3.sum().EVENT
    d3["DIST_NON_EVENT"] = d3.NONEVENT/d3.sum().NONEVENT
    d3["WOE"] = np.log(d3.DIST_EVENT/d3.DIST_NON_EVENT)
    d3["IV"] = (d3.DIST_EVENT-d3.DIST_NON_EVENT)*np.log(d3.DIST_EVENT/d3.DIST_NON_EVENT)
    d3["VAR_NAME"] = "VAR"
    d3 = d3[['VAR_NAME','MIN_VALUE', 'MAX_VALUE', 'COUNT', 'EVENT', 'EVENT_RATE', 'NONEVENT', 'NON_EVENT_RATE', 'DIST_EVENT','DIST_NON_EVENT','WOE', 'IV']]      
    d3 = d3.replace([np.inf, -np.inf], 0)
    d3.IV = d3.IV.sum()
    d3 = d3.reset_index(drop=True)
    
    return(d3)

def data_vars(df1, target):
    
    stack = traceback.extract_stack()
    filename, lineno, function_name, code = stack[-2]
    vars_name = re.compile(r'\((.*?)\).*$').search(code).groups()[0]
    final = (re.findall(r"[\w']+", vars_name))[-1]
    
    x = df1.dtypes.index
    count = -1
    
    for i in x:
        if i.upper() not in (final.upper()):
            if np.issubdtype(df1[i], np.number) and len(Series.unique(df1[i])) > 2:
                conv = mono_bin(target, df1[i])
                conv["VAR_NAME"] = i
                count = count + 1
            else:
                conv = char_bin(target, df1[i])
                conv["VAR_NAME"] = i            
                count = count + 1
                
            if count == 0:
                iv_df = conv
            else:
                iv_df = iv_df.append(conv,ignore_index=True)
    
    iv = pd.DataFrame({'IV':iv_df.groupby('VAR_NAME').IV.max()})
    iv = iv.reset_index()
    return(iv_df,iv)


#final_iv, IV = data_vars(df[df.columns.difference(['ClassInd'])],df.ClassInd)
final_iv, IV = data_vars(final_df[final_df.columns.difference([target_var])],final_df[target_var])
IV = IV.rename(columns={'VAR_NAME':'index'})
IV.sort_values(['IV'],ascending=0)

'''-----------------------------------------
2. Variable Importance using Random Forest(RandomforestClassifier)
------------------------------------------'''
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()

clf.fit(features,labels)

preds = clf.predict(features)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(preds,labels)
print('Random Forest model accuracy',accuracy)

from pandas import DataFrame
VI = DataFrame(clf.feature_importances_, columns = ["RF"], index=features.columns)

VI = VI.reset_index()
VI=VI.sort_values(['RF'],ascending=0)

'''-----------------------------------------
3 Recursive Feature Elimination
------------------------------------------'''
from sklearn.svm import LinearSVC
from sklearn.feature_selection import RFE

model = LinearSVC()
rfe = RFE(model, total_feature)
fit = rfe.fit(features, labels)

from pandas import DataFrame
Selected = DataFrame(rfe.support_, columns = ["RFE"], index=features.columns)
Selected = Selected.reset_index()
#print(rfe.ranking_)
'''-----------------------------------------
4 Variable Importance using Extratrees Classifier
------------------------------------------'''
from sklearn.ensemble import ExtraTreesClassifier

model = ExtraTreesClassifier()
model.fit(features, labels)

print(model.feature_importances_)
from pandas import DataFrame
FI = DataFrame(model.feature_importances_, columns = ["Extratrees"], index=features.columns)

FI = FI.reset_index()
FI.sort_values(['Extratrees'],ascending=0)


'''-----------------------------------------
5 Chi Square
------------------------------------------'''
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

model = SelectKBest(score_func=chi2, k='all')
fit = model.fit(features.abs(), labels)

from pandas import DataFrame
pd.options.display.float_format = '{:.2f}'.format
chi_sq = DataFrame(fit.scores_, columns = ["Chi_Square"], index=features.columns)

chi_sq = chi_sq.reset_index()

chi_sq=chi_sq.sort_values('Chi_Square',ascending=0)

'''-----------------------------------------
6 L1 feature selection
------------------------------------------'''
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(features, labels)
model = SelectFromModel(lsvc,prefit=True)

from pandas import DataFrame
l1 = DataFrame(model.get_support(), columns = ["L1"], index=features.columns)
l1 = l1.reset_index()


'''-----------------------------------------
7 Combine all together
------------------------------------------'''
from functools import reduce
dfs = [IV, VI, Selected, FI, chi_sq, l1]
final_results = reduce(lambda left,right: pd.merge(left,right,on='index'), dfs)

'''-----------------------------------------

8 Vote each variable
------------------------------------------'''
columns = [ 'IV', 'RF', 'Extratrees', 'Chi_Square']

score_table = pd.DataFrame({},[])
score_table['index'] = final_results['index']

for i in columns:
    score_table[i] = final_results['index'].isin(list(final_results.nlargest(5,i)['index'])).astype(int)
    
score_table['RFE'] = final_results['RFE'].astype(int)
score_table['L1'] = final_results['L1'].astype(int)
score_table['final_score'] = score_table.sum(axis=1)
score_table=score_table.sort_values('final_score',ascending=0)

'''-----------------------------------------

9 Multicollinearity
------------------------------------------'''
from statsmodels.stats.outliers_influence import variance_inflation_factor
def calculate_vif(features):
    vif = pd.DataFrame()
    vif["Features"] = features.columns
    vif["VIF"] = [variance_inflation_factor(features.values, i) for i in range(features.shape[1])]    
    return(vif)
    
features1 = features[list(score_table[score_table['final_score'] >= 2]['index'])]
x=features1.dtypes
features.dtypes
vif = calculate_vif(features1)
while vif['VIF'][vif['VIF'] > 10].any():
    remove = vif.sort_values('VIF',ascending=0)['Features'][:1]
    print('feature removed: '+ str(remove) )
    features1.drop(remove,axis=1,inplace=True)
    vif = calculate_vif(features1)
    
#list(vif['Features'])

Final_features=list(vif['Features'])
Final_vars = list(vif['Features']) +[target_var]




'''-----------------------------------------

10. PRINT ALL TABLES
------------------------------------------'''

'''#-------tRAINING DATA SUMMARY-----'''
print('*'*15 + 'SUMMARY AND DATATYPE ' + '*'*15)
print('Count of class :\n', df[target_var].value_counts())
print('\n Ratio of class :\n',df[target_var].value_counts()/len(df))
print('\n Data Type of the variables :\n',df.dtypes)
print('Sample Training Data \n',df.head())
print('*'*50)


''' # 3. # Get Numeric/non_neumeric Features ---'''
print('*'*15 + 'Neumeric/Non-Numeric DF shape  ' + '*'*15)
print('Shape of Neumeric DF : ', Numeric_data.shape)
print('Shape of Non-Neumeric DF : ', non_neumeric_df.shape)
print('\n Sample Neumeric DF \n',Numeric_data.head())
print('Sample Non-Neumeric DF \n',non_neumeric_df.head())
print('*'*50)

''' 4. Null Value Imputation ---'''
print('*'*15 + 'NULL VALUE TABLE BEFORE and AFTER  MEAN IMPUTATION ' + '*'*15)
print('Feature wise Numm value %  before Imputation: \n',Mis_Valu_befor_Imp)
print('Feature wise Numm value %  after  Imputation: \n',Null_val_after_imp)
print('*'*50)

'''#-------1.WOE and IV--------------'''
print('*'*20 + 'WOE and IV ' + '*'*20)
print('Weight of Evidence  :\n', final_iv)
print('Importent Variavles: \n',IV)
print('*'*50)

'''2 Variable Importance using Random Forest(RandomforestClassifier)'''

print('*'*20 + 'Variable Importance using Random Forest ' + '*'*20)
print('Variable Importance :\n', VI)
print('*'*50)

'''3 Recursive Feature Elimination'''

print('*'*20 + 'Recursive Feature Elimination ' + '*'*20)
print('RFE :\n', Selected)
print('*'*50)

'''4 Variable Importance using Extratrees Classifier'''
print('*'*20 + 'Variable Importance using Extratrees Classifier ' + '*'*20)
print('FI :\n', FI)
print('*'*50)

'''5 Chi Square'''
print('*'*20 + 'Univariate feature selection-Chi-Square ' + '*'*20)
print('Chi-Square  :\n', chi_sq )
print('*'*50)

''' 6 L1 feature selection'''
print('*'*20 + 'L1-based feature selection ' + '*'*20)
print('Chi-Square  :\n', l1 )
print('*'*50)


''' 7 Combine all together'''
print('*'*20 + 'Combine all method ' + '*'*20)
print('Combined Table :\n', final_results )
print('*'*50)


''' 8 Vote each variable'''
print('*'*20 + 'Each Variable Voting Score ' + '*'*20)
print('Voting Table :\n', score_table )
print('*'*50)



''' 9 Multicollinearity-using variance inflasion factor'''
print('*'*20 + 'Each Variable VIF Score ' + '*'*20)
print('VIF Table :\n', vif )
print('*'*50)

'''#------10 Final Features -----'''
print('*'*20 + 'FINAL FEATURES ' + '*'*20)
print('final_vars :\n', Final_vars)
print('FINAL FEATURE :\n', Final_features)
print('*'*70)

''' 
####  ======
'''

'''-----------------------------------------

10. SAVE ALL FILES
------------------------------------------'''


#========1 WOE AND Information Value
#*** It is applicable for Binary Classification
today = datetime.now()
mkdir=os.getcwd()+"//1_WOE_AND_Information_Value//"+today.strftime("%Y-%m-%d-%H%M%S")
if not os.path.exists(mkdir):
    os.makedirs(mkdir)
    
final_iv.to_csv(mkdir +'\\WOE_and_IV.csv')
IV.to_csv(mkdir +'\\IV.csv')


#========9.2 Variable Importance using Random Forest

today = datetime.now()
mkdir=os.getcwd()+"//2_Var_Imp_Table_Using_RF//"+today.strftime("%Y-%m-%d-%H%M%S")
if not os.path.exists(mkdir):
    os.makedirs(mkdir)
    
VI.to_csv(mkdir +'\Var_Imp.csv')

#========9.3 Recursive Feature Elimination
mkdir=os.getcwd()+"//3_Recursive_Feature_Elimination"
if not os.path.exists(mkdir):
    os.makedirs(mkdir)
Selected.to_csv(mkdir+'\\RFE.csv')

#========4 Variable Importance  USING Tree-based feature selection
today = datetime.now()
mkdir=os.getcwd()+"//4_Var_Imp_using_Extratrees_Classifier//"+today.strftime("%Y-%m-%d-%H%M%S")
if not os.path.exists(mkdir):
    os.makedirs(mkdir)
FI.to_csv(mkdir +'\\FI.csv')

#========5 Chi Square

today = datetime.now()
mkdir=os.getcwd()+"//5_Chi_Square//"+today.strftime("%Y-%m-%d-%H%M%S")
if not os.path.exists(mkdir):
    os.makedirs(mkdir)
chi_sq.to_csv(mkdir+'\\chi_sq.csv')

#========6 L1 feature selection

today = datetime.now()
mkdir=os.getcwd()+"//6_L1_feature_selection//"+today.strftime("%Y-%m-%d-%H%M%S")
if not os.path.exists(mkdir):
    os.makedirs(mkdir)
l1.to_csv(mkdir+'\\l1.csv')

#=======9.7 Combine all together
today = datetime.now()
mkdir=os.getcwd()+"//7_Combine_all_together//"+today.strftime("%Y-%m-%d-%H%M%S")
if not os.path.exists(mkdir):
    os.makedirs(mkdir)
final_results.to_csv(mkdir+'\\final_results_combind.csv')

#======9.8 Vote each variable
today = datetime.now()
mkdir=os.getcwd()+"//8_Vote_each_variable//"+today.strftime("%Y-%m-%d-%H%M%S")
if not os.path.exists(mkdir):
    os.makedirs(mkdir)
score_table.to_csv(mkdir+'\\Voting_Table.csv')

#======9.9 Multicollinearity-using variance inflasion factor
today = datetime.now()
mkdir=os.getcwd()+"//9_Multicolearity_VIF//"+today.strftime("%Y-%m-%d-%H%M%S")
if not os.path.exists(mkdir):
    os.makedirs(mkdir)
score_table.to_csv(mkdir+'\\VIF.csv')

#======9.10 Multicollinearity-using variance inflasion factor
today = datetime.now()
mkdir=os.getcwd()+"//10_Final_Features//"+today.strftime("%Y-%m-%d-%H%M%S")
if not os.path.exists(mkdir):
    os.makedirs(mkdir)
    
final_vars1=pd.DataFrame(Final_vars)
final_vars1.to_csv(mkdir+'\\Final_vars.csv')

final_features1=pd.DataFrame(Final_features)
final_features1.to_csv(mkdir+'\\Final_features.csv')
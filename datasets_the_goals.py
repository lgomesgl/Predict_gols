# datasets options
'''
    features --> Time, HomeTeam, AwayTeam, Referee, Home Time Shots on Target,
    Away Team Shots on Target, Kick efficiency(last season) --> Mean shots on target/Mean gols
    
    1 option --> Predict the total gols (Regresion)
    
    2 option --> Predict if total gols +- 2.5 (Classifier)
    
    -- Concatenate 2 years of the championship
    -- Focus in 1.5 classifier 
    -- Create a new feature 
    
    
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from analyse_the_goals import df_gols, df_features, df_home_features, df_away_features
# , df_gols_2, df_features_2,df_home_features_2, df_away_features_2

def data_1_option(data):
    return data[['DTO','Time','HomeTeam','AwayTeam','Referee','TG']]

def data_2_option(data, data_home_features, data_away_features):
    
    home_teams = data['HomeTeam'].values
    away_teams = data['AwayTeam'].values
    
    _kick_eff_home, _kick_eff_home_opp = [], []
    _kick_eff_away, _kick_eff_away_opp = [], []
    
    for team in home_teams:    
        kick_eff = data_home_features[(data_home_features['team'] == team)]['Kick_eff'].values[0]
        kick_eff_opp = data_home_features[(data_home_features['team'] == team)]['Kick_eff_opp'].values[0]
        
        _kick_eff_home.append(kick_eff), _kick_eff_home_opp.append(kick_eff_opp)
        
    for team in away_teams:
        kick_eff = data_away_features[(data_away_features['team'] == team)]['Kick_eff'].values[0]
        kick_eff_opp = data_away_features[(data_away_features['team'] == team)]['Kick_eff_opp'].values[0]
        
        _kick_eff_away.append(kick_eff), _kick_eff_away_opp.append(kick_eff_opp)

    data['HKE'] = _kick_eff_home
    data['HKEP'] = _kick_eff_home_opp
    data['AHE'] = _kick_eff_away
    data['AKEP'] = _kick_eff_away_opp

    def classifier_gols(value):
        if value < 2.5:
            return 0.0
        else:
            return 1.0
    
    data['TG'] = data['TG'].apply(lambda value: classifier_gols(value))
    
    return data[['Time','HomeTeam','AwayTeam','Referee','HKE','HKEP','AHE','AKEP','TG']]

def concatenated_datas(data_1,data_2):
    return pd.concat([data_1,data_2])


# df_1 = data_1_option(df_nf_c)
df_2 = data_2_option(df_features, df_home_features, df_away_features)
# df_2_2 = data_2_option(df_features_2, df_home_features_2, df_away_features_2)

# data_plus = concatenated_datas(df_2,df_2_2)
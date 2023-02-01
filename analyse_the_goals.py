# predict football -- 20/12/22 -- analyse the data
'''
    FTHG: Full time home team goals
    FTAG: Full time away team goals
    FTR: Full time results
    
    HTHG: Half time home team goals
    HTAG: Half time away team goals
    HTR: Half time results
    
'''
# EDA & Visualizer
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns

def read_the_data(data, delimiter):
    return pd.read_csv(data, delimiter=delimiter)

def data_for_goals(data, st_column, lt_column):
    return data.iloc[:, st_column:lt_column] 

def new_features_1(data):
    data['TG'] = data['FTHG'] + data['FTAG'] # total gols
    data['HTG'] = data['HTHG'] + data['HTAG'] # total gols in half time
    
    data['D_HG'] = data['FTHG'] - data['HTHG'] # difference by half home gols
    data['D_AG'] = data['FTAG'] - data['HTAG'] # difference by half away gols
    data['D_HTG_TG'] = data['TG'] - data['HTG'] # difference by total gols
    
    data['Date'] = pd.to_datetime(data['Date']) # difference between the date of the game and the last day of championship
    last_day = data.shape[0] - 1
    data['DTO'] = data['Date'][last_day] - data['Date']
    data['DTO'] = data['DTO'].astype(str)
    data['DTO'] = data['DTO'].apply(lambda row: row.split()[0])
    
    def _rounds(data): # rounds 
        new_column = np.array([])
        for i in range(1, int(data.shape[0]/10) + 1):
            new_round = np.full((1,10), float(i))
            new_column = np.concatenate((new_column, new_round), axis = None)
        return new_column
    
    new_column = _rounds(data)
    # data['RD'] = new_column
    
    return data

def DTO_conditional(data):
    return data[(data['DTO'] >= '0')]

# take the mean of the goals, shots on target, kick efficienty by each team in home and away games.
def new_features_2(data,side,side_letter):
    home_teams = sorted(data[side].unique())
    
    teams = []
    _shoots_on_target = []   
    _mean_gols = [] 
    _kick_efficienty = []
    
    _shoots_on_target_opp = []
    _mean_gols_opp = []
    _kick_efficienty_opp = []
    
    if side_letter == 'H':
        opp_letter = 'A'
    else: 
        opp_letter = 'H'

    for team in home_teams:
        df = data[(data[side] == team)]
        shoots_on_target = df['%sST' % side_letter].describe()[1]
        mean_gols = df['FT%sG' % side_letter].describe()[1]
        
        shoots_on_target_opp = df['%sST' % opp_letter].describe()[1]
        mean_gols_opp = df['FT%sG' % opp_letter].describe()[1]
        
        teams.append(team)
        
        _shoots_on_target.append(shoots_on_target)
        _mean_gols.append(mean_gols)
        _kick_efficienty.append(shoots_on_target/mean_gols)
        
        _shoots_on_target_opp.append(shoots_on_target_opp)
        _mean_gols_opp.append(mean_gols_opp)
        _kick_efficienty_opp.append(shoots_on_target_opp/mean_gols_opp)
            
    df_features = pd.DataFrame({'team':teams, 
                                
                                'mean_gols':_mean_gols, 
                                'shoots_on_target':_shoots_on_target, 
                                'Kick_eff':_kick_efficienty,
                                
                                'mean_gols_opp':_mean_gols_opp, 
                                'shoots_on_target_opp':_shoots_on_target_opp, 
                                'Kick_eff_opp':_kick_efficienty_opp})
    return df_features

def plot_mean_gols(data, x_label, options):
    while options == True:
        sns.displot(data[x_label])
        plt.show()
    
    for i, value in enumerate(['FTHG','FTAG','HTHG','HTAG']):
        plt.subplot(2,2,i+1)
        sns.barplot(data=data, x = x_label, y = value)
        if i < 2:
            plt.xticks([])
        plt.xticks(rotation=90)
    plt.show()
    
    sns.barplot(data=data, x=x_label, y = 'TG')
    plt.show()
   
def heatmap(data):
    data_corr = data.corr()
    sns.heatmap(data=data_corr, annot=True, cmap='mako')
    plt.show()   

data_1 = read_the_data('Premier_21_22.csv', ',')
df_gols = data_for_goals(data_1, 1, 24)
df_features = new_features_1(df_gols)
df_home_features = new_features_2(df_gols,'HomeTeam','H')
df_away_features = new_features_2(df_gols,'AwayTeam','A')

# data_2 = read_the_data('Premier_22_23.csv', ',')
# df_gols_2 = data_for_goals(data_1, 1, 24)
# df_features_2 = new_features_1(df_gols_2)
# df_home_features_2 = new_features_2(df_gols_2,'HomeTeam','H')
# df_away_features_2 = new_features_2(df_gols_2,'AwayTeam','A')





# df_nf_c = DTO_conditional(df_nf)

# graphics:
# plot_mean_gols(df_nf, 'Time', False)
# heatmap(df_nf_c)
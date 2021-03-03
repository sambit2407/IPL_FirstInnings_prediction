from django.db import models

# Create your models here.
# !/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

data = pd.read_csv("C://Users//user//PycharmProjects//djangoModel//ipl.csv")
data.head()
data.shape

# In[2]:


# DataCleaning
col_to_rem = ['mid', 'batsman', 'bowler', 'striker', 'non-striker']
data.drop(columns=col_to_rem, axis=0, inplace=True)
data.head()
data.shape

# In[3]:


data['bat_team'].unique()

# In[4]:


recent_team = ['Kolkata Knight Riders', 'Chennai Super Kings', 'Rajasthan Royals', 'Mumbai Indians', 'Kings XI Punjab',
               'Royal Challengers Bangalore', 'Delhi Daredevils', 'Sunrisers Hyderabad']

# In[5]:


data = data[data['bat_team'].isin(recent_team) & data['bowl_team'].isin(recent_team)]

# In[6]:


data['bat_team'].unique()

# In[7]:


data['bowl_team'].unique()

# In[8]:


# Removing First 5 OVer match
data = data[data['overs'] >= 5]

# In[9]:


data.head()
data.shape

# In[10]:


# converting date col from string type
data['date'] = pd.to_datetime(data['date'])

# In[11]:


data.head()

# In[94]:


data['venue'].unique()

# In[13]:


# top_20_venue=data.venue.value_counts().sort_values(ascending=False).head(20).index
# top_20_venue


# In[14]:


# def one_hot_topx(dataset,var,topx):
# for label in top_20_venue:
# dataset[var+'_'+label]=np.where(data[var]==label,1,0)


# In[15]:


import numpy as np

# one_hot_topx(data,var='venue',topx=top_20_venue)


# data.drop(columns='venue',axis=1,inplace=True)


# In[16]:


data.shape

# In[17]:


data['bat_team'].nunique()

# In[18]:


data_encoded = pd.get_dummies(data=data)

# In[19]:


data.head()

# In[20]:


data_dep = data.copy()

# In[21]:


data_dep.drop(columns='date', inplace=True)

# In[22]:


print(data_dep)

# In[23]:


data_encoded.head()

# In[24]:


# Rearranging the cols
data_encoded.columns

# In[25]:


data_encoded = data_encoded[['venue_Barabati Stadium', 'venue_Brabourne Stadium',
                             'venue_Buffalo Park', 'venue_De Beers Diamond Oval',
                             'venue_Dr DY Patil Sports Academy',
                             'venue_Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium',
                             'venue_Dubai International Cricket Stadium', 'venue_Eden Gardens',
                             'venue_Feroz Shah Kotla',
                             'venue_Himachal Pradesh Cricket Association Stadium',
                             'venue_Holkar Cricket Stadium',
                             'venue_JSCA International Stadium Complex', 'venue_Kingsmead',
                             'venue_M Chinnaswamy Stadium', 'venue_MA Chidambaram Stadium, Chepauk',
                             'venue_Maharashtra Cricket Association Stadium',
                             'venue_New Wanderers Stadium', 'venue_Newlands',
                             'venue_OUTsurance Oval',
                             'venue_Punjab Cricket Association IS Bindra Stadium, Mohali',
                             'venue_Punjab Cricket Association Stadium, Mohali',
                             'venue_Rajiv Gandhi International Stadium, Uppal',
                             'venue_Sardar Patel Stadium, Motera', 'venue_Sawai Mansingh Stadium',
                             'venue_Shaheed Veer Narayan Singh International Stadium',
                             'venue_Sharjah Cricket Stadium', 'venue_Sheikh Zayed Stadium',
                             "venue_St George's Park", 'venue_Subrata Roy Sahara Stadium',
                             'venue_SuperSport Park', 'venue_Wankhede Stadium',
                             'bat_team_Chennai Super Kings', 'bat_team_Delhi Daredevils',
                             'bat_team_Kings XI Punjab', 'bat_team_Kolkata Knight Riders',
                             'bat_team_Mumbai Indians', 'bat_team_Rajasthan Royals',
                             'bat_team_Royal Challengers Bangalore', 'bat_team_Sunrisers Hyderabad',
                             'bowl_team_Chennai Super Kings', 'bowl_team_Delhi Daredevils',
                             'bowl_team_Kings XI Punjab', 'bowl_team_Kolkata Knight Riders',
                             'bowl_team_Mumbai Indians', 'bowl_team_Rajasthan Royals',
                             'bowl_team_Royal Challengers Bangalore',
                             'bowl_team_Sunrisers Hyderabad', 'date', 'runs', 'wickets', 'overs', 'runs_last_5',
                             'wickets_last_5',
                             'total']]

# In[26]:


data_encoded.head()

# In[27]:


# Spliting Data

X_train = data_encoded.drop(labels='total', axis=1)[data_encoded['date'].dt.year <= 2016]

X_test = data_encoded.drop(labels='total', axis=1)[data_encoded['date'].dt.year >= 2017]

# In[28]:


X_test.head()

# In[29]:


y_train = data_encoded[data_encoded['date'].dt.year <= 2016]['total'].values
y_test = data_encoded[data_encoded['date'].dt.year >= 2017]['total'].values

# In[30]:


X_test.columns

# In[31]:


# droping date col as it is not going to help in ML building
X_train.drop(labels='date', axis=True, inplace=True)
X_test.drop(labels='date', axis=True, inplace=True)

# In[ ]:


# # XGboost regressor

# In[ ]:


# In[32]:


# Hyperparameter Tuning
params = {
    'max_depth': [1, 3, 5, 7, 10],
    'learning_rate': [0.05, 0.1, 0.2, 0.25, 0.3, 0.4],
    'min_child_weight': [1, 3, 5, 7],
    'colsample_bytree': [0.3, 0.4, 0.5, 0.6],
    'gamma': [0.1, 0.2, 0.3, 0.4, 0.5]
}

# In[33]:


from xgboost import XGBRegressor

regressor = XGBRegressor()

# In[34]:


from sklearn.model_selection import RandomizedSearchCV

random_search = RandomizedSearchCV(regressor, param_distributions=params, n_iter=5, cv=5, n_jobs=-1, verbose=3)

# In[35]:


X_train.columns

# In[36]:


random_search.fit(X_train, y_train)

# In[37]:


random_search.best_estimator_

# In[38]:


regressor = XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                         colsample_bynode=1, colsample_bytree=0.6, gamma=0.1, gpu_id=-1,
                         importance_type='gain', interaction_constraints='',
                         learning_rate=0.2, max_delta_step=0, max_depth=3,
                         min_child_weight=5, monotone_constraints='()',
                         n_estimators=100, n_jobs=0, num_parallel_tree=1, random_state=0,
                         reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
                         tree_method='exact', validate_parameters=1, verbosity=None)

# In[39]:


X_train.values

# In[40]:


y_train

# In[41]:


regressor.fit(X_train, y_train)

# In[42]:


from sklearn.model_selection import cross_val_score

score = cross_val_score(regressor, X_train, y_train, cv=10)

# In[43]:


score

# In[44]:


len(X_test.columns)

# In[45]:


prediction = regressor.predict(X_test)

# In[46]:


predData = pd.DataFrame({'predict': prediction, 'actual': y_test})
predData

# In[47]:


import seaborn as sns

sns.distplot(prediction - y_test)

# In[48]:


from sklearn import metrics

metrics.median_absolute_error(prediction, y_test)

# In[49]:


import pickle

with open('First_inn_Model.pkl', 'wb') as fileWriteStream:
    pickle.dump(regressor, fileWriteStream)

# # Creating API for Deployment
#
#

# In[71]:


testData = pd.DataFrame(
    data=[["Dubai International Cricket Stadium", 'Chennai Super Kings', "Kolkata Knight Riders", 20, 3, 3, 3, 3]],
    columns=['venue', 'bat_team', 'bowl_team', 'runs', 'wickets',
             'overs', 'runs_last_5', 'wickets_last_5'])


# In[89]:


def predictFun(InputMatchStatus):
    import pandas as pd
    Num_Inputs = InputMatchStatus.shape[0]
    # print(Num_Inputs)
    # print(InputMatchStatus)

    # Making sure the input data has same columns as it was used for training the model
    # Also, if standardization/normalization was done, then same must be done for new input

    # Appending the new data with the Training data
    # DataForML=pd.read_pickle('DataForML.pkl')
    # print(data_dep)
    InputMatchStatus = InputMatchStatus.append(data_dep)
    # print(InputMatchStatus)

    # Treating the Ordinal variable first
    # InputLoanDetails['employ'].replace({'A71':1, 'A72':2,'A73':3, 'A74':4,'A75':5 }, inplace=True)

    # Generating dummy variables for rest of the nominal variables
    InputMatchStatus = pd.get_dummies(InputMatchStatus)
    # print(InputMatchStatus['venue_Eden Gardens'])

    # Maintaining the same order of columns as it was during the model training
    Predictors = ['venue_Barabati Stadium', 'venue_Brabourne Stadium',
                  'venue_Buffalo Park', 'venue_De Beers Diamond Oval',
                  'venue_Dr DY Patil Sports Academy',
                  'venue_Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium',
                  'venue_Dubai International Cricket Stadium', 'venue_Eden Gardens',
                  'venue_Feroz Shah Kotla',
                  'venue_Himachal Pradesh Cricket Association Stadium',
                  'venue_Holkar Cricket Stadium',
                  'venue_JSCA International Stadium Complex', 'venue_Kingsmead',
                  'venue_M Chinnaswamy Stadium', 'venue_MA Chidambaram Stadium, Chepauk',
                  'venue_Maharashtra Cricket Association Stadium',
                  'venue_New Wanderers Stadium', 'venue_Newlands',
                  'venue_OUTsurance Oval',
                  'venue_Punjab Cricket Association IS Bindra Stadium, Mohali',
                  'venue_Punjab Cricket Association Stadium, Mohali',
                  'venue_Rajiv Gandhi International Stadium, Uppal',
                  'venue_Sardar Patel Stadium, Motera', 'venue_Sawai Mansingh Stadium',
                  'venue_Shaheed Veer Narayan Singh International Stadium',
                  'venue_Sharjah Cricket Stadium', 'venue_Sheikh Zayed Stadium',
                  "venue_St George's Park", 'venue_Subrata Roy Sahara Stadium',
                  'venue_SuperSport Park', 'venue_Wankhede Stadium',
                  'bat_team_Chennai Super Kings', 'bat_team_Delhi Daredevils',
                  'bat_team_Kings XI Punjab', 'bat_team_Kolkata Knight Riders',
                  'bat_team_Mumbai Indians', 'bat_team_Rajasthan Royals',
                  'bat_team_Royal Challengers Bangalore', 'bat_team_Sunrisers Hyderabad',
                  'bowl_team_Chennai Super Kings', 'bowl_team_Delhi Daredevils',
                  'bowl_team_Kings XI Punjab', 'bowl_team_Kolkata Knight Riders',
                  'bowl_team_Mumbai Indians', 'bowl_team_Rajasthan Royals',
                  'bowl_team_Royal Challengers Bangalore',
                  'bowl_team_Sunrisers Hyderabad', 'runs', 'wickets', 'overs',
                  'runs_last_5', 'wickets_last_5']

    # Generating the input values to the model
    X = InputMatchStatus[Predictors][0:Num_Inputs]
    # print(X)
    # Generating the standardized values of X since it was done while model training also
    # X=PredictorScalerFit.transform(X)

    # Loading the Function from pickle file
    import pickle
    with open('First_inn_Model.pkl', 'rb') as fileReadStream:
        regressor = pickle.load(fileReadStream)
        # Don't forget to close the filestream!
        fileReadStream.close()
    # print(X.columns)

    # Genrating Predictions
    Prediction = int(regressor.predict(X))
    # print(Prediction)
    # PredictedStatus=pd.DataFrame(Prediction, columns=['Predicted Status'])

    # upper_val=Prediction + 10
    return (Prediction)


# In[90]:


# In[91]:


# Creating the function which can take loan inputs and perform prediction
def FirstInnPrediction(inp_venue, inp_bat_team, inp_bowl_team, inp_runs,
                       inp_overs, inp_wickets, inp_runs_last_5,
                       inp_wickets_last_5):
    SampleInputData = pd.DataFrame(
        data=[[inp_venue, inp_bat_team, inp_bowl_team, inp_runs,
               inp_wickets, inp_overs, inp_runs_last_5,
               inp_wickets_last_5]],
        columns=['venue', 'bat_team', 'bowl_team', 'runs', 'wickets',
                 'overs', 'runs_last_5', 'wickets_last_5'])
    # print(SampleInputData.shape[0])

    # Calling the function defined above using the input parameters
    Predictions = predictFun(InputMatchStatus=SampleInputData)

    # Returning the predicted loan status
    return (Predictions)







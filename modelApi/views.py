from django.http import HttpResponse
from django.shortcuts import render
import pandas as pd


data = pd.read_csv("ipl.csv")
col_to_rem = ['mid', 'batsman', 'bowler', 'striker', 'non-striker']
data.drop(columns=col_to_rem, axis=0, inplace=True)
recent_team = ['Kolkata Knight Riders', 'Chennai Super Kings', 'Rajasthan Royals', 'Mumbai Indians', 'Kings XI Punjab',
               'Royal Challengers Bangalore', 'Delhi Daredevils', 'Sunrisers Hyderabad']
data = data[data['bat_team'].isin(recent_team) & data['bowl_team'].isin(recent_team)]
data = data[data['overs'] >= 5]
data_dep = data.copy()
data_dep.drop(columns='date', inplace=True)



def home(request):
    return render(request,"home.html")
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
                  'bowl_team_Mumbai Indians','bowl_team_Rajasthan Royals',
                  'bowl_team_Royal Challengers Bangalore',
                  'bowl_team_Sunrisers Hyderabad','runs', 'wickets', 'overs',
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

def predict(request):
    inp_venue = request.GET.get('Chose the Venue')
    inp_bat_team = request.GET.get('Chose the Batting_Team')
    inp_bowl_team = request.GET.get('Chose the Bowling_Team')
    inp_runs = int(request.GET.get('Runs'))
    inp_overs = float(request.GET.get('Overs'))
    inp_wickets = int(request.GET.get('Wickets'))
    inp_runs_last_5 = int(request.GET.get('Runs Last 5 Overs'))
    inp_wickets_last_5 = int(request.GET.get('Wickets Last 5 Overs'))
    #print(inp_venue,inp_overs,inp_runs,inp_bat_team,inp_bowl_team,inp_runs_last_5,inp_wickets,inp_wickets_last_5)
    if inp_bat_team==inp_bowl_team:
        return render(request,"error.html")
    if inp_runs_last_5 > inp_runs:
        return render(request,"runError.html")
    if inp_wickets_last_5 > inp_wickets:
        return render(request,"wicketError.html")

    # Calling the funtion to get firstInn Prediction status
    prediction_from_api = FirstInnPrediction(
        inp_venue, inp_bat_team, inp_bowl_team, inp_runs,
        inp_overs, inp_wickets, inp_runs_last_5,
        inp_wickets_last_5)
    lower_val = prediction_from_api - 5
    upper_val = prediction_from_api + 5
    # print(lower_val)

    #return (inp_bat_team + ' first inn predicted scorewould be : ' + str(lower_val) + ' to ' + str(upper_val))

    

    return render(request,"predict.html",{'lower_val':lower_val,'upper_val':upper_val,'inp_bat_team':inp_bat_team})

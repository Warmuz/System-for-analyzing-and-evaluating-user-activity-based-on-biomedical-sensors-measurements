import dash
import dash_core_components as dcc
import dash_html_components as html

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import math
from numpy import random
import statistics



df = pd.read_csv('C:/Users/Mateusz/Desktop/XaiomiOpaska1/HEARTRATE_AUTO/HR_auto.csv')


def timeconverter(time):
    (hour, minute) = time.split(':')
    time = int(hour) * 60 + int(minute)
    return time


def trainingimpulse(data, duration):
    min = df['heartRate'].min()
    max = df['heartRate'].max()
    TRIMP = []

    for i in range(0, duration):
        particularTrimp = ((data.iloc[i] - min) / (max - min)) * 0.64 * math.exp(1.92 * ((data.iloc[i] - min) / (max - min)))

        TRIMP.append(particularTrimp)

    return sum(TRIMP)


def cp(time):
    score = []
    for i in range(0, len(time)):
        temp = 294.7 * math.log(11.9 / (time[i] - 3.1))
        score.append(temp)
    return score


def filtration(interval, condition):
    filtrated = pd.DataFrame()

    HRmaxall = df['heartRate'].max()
    firstZone = 0.6 * HRmaxall
    secondZone = 0.7 * HRmaxall
    thirdZone = 0.8 * HRmaxall
    fourthZone = 0.9 * HRmaxall
    fifthZone = HRmaxall

    # fIltration by heart rate zones
    HRtraining_all = df.loc[(df['heartRate'] > firstZone)]
    length = len(HRtraining_all)

    # filtration by checking the interval beetween heart rate
    for i in range(0, length - interval):
        temp = timeconverter(HRtraining_all['time'].iloc[i + 1]) - timeconverter(HRtraining_all['time'].iloc[i])

        if (temp <= condition):
            filtrated = pd.concat([filtrated, HRtraining_all.iloc[[i]]])

    # filtering data by each day
    firstdate = filtrated['date'].iloc[0]
    lastday = filtrated['date'].iloc[-1]
    end = df['date'].iloc[-1]
    date_time_obj1 = datetime.datetime.strptime(firstdate, '%Y-%m-%d')
    date_time_obj2 = datetime.datetime.strptime(end, '%Y-%m-%d')

    delta = datetime.timedelta(1)

    startTime = date_time_obj1
    trimp = []
    date = []

    while startTime <= date_time_obj2:
        startstring = startTime
        startstring = startstring.strftime('%Y-%m-%d')

        particularDay = filtrated.loc[filtrated['date'] == startstring]
        copy = pd.DataFrame()

        if len(particularDay) > 10:

            for i in range(0, len(particularDay) - 1):
                temp = particularDay.index[i + 1] - particularDay.index[i]
                copy = pd.concat([copy, particularDay.iloc[[i]]])

                if temp > 30:
                    break

            # duration = timeconverter(copy['time'].iloc[-1])-timeconverter(copy['time'].iloc[0])

            trimp.append(trainingimpulse(copy['heartRate'], len(copy['heartRate'])))

        else:
            trimp.append(0)

        startTime = startTime + delta
        date.append(startstring)

    frame = {'date': date, 'trimp': trimp}

    niceframe = pd.DataFrame(data=frame)

    #plt.bar(date, trimp, color='blue')
    #plt.xticks(rotation=90)

    return niceframe

def banistermodel(alldata, t1, t2, p0, k1, k2):
    data = alldata['trimp']

    fitness = np.zeros([len(data)], dtype=float)
    fatigue = np.zeros([len(data)], dtype=float)
    performance = np.zeros([len(data)], dtype=float)
    performance[0] = p0

    for i in range(1, len(data)):
        fitness[i] = data[i] + fitness[i - 1] * math.exp(-(len(data) - i) / t1)
        fatigue[i] = data[i] + fatigue[i - 1] * math.exp(-(len(data) - i) / t2)
        performance[i] = p0 + k1 * fitness[i] - k2 * fatigue[i]

    frame = {'date': alldata['date'], 'performance': performance, 'fitness': fitness, 'fatigue': fatigue}
    all = pd.DataFrame(data=frame)

    return all


def fit(data):
    date = ['2021-03-23', '2021-04-14', '2021-04-21', '2021-05-03', '2021-05-17', '2021-06-07']
    trimp = np.zeros([len(date)], dtype=float)
    position = np.zeros([len(date)], dtype=float)
    zeros = np.zeros([len(data)], dtype=float)
    maks = np.zeros([len(date)], dtype=float)

    for i in range(0, len(date)):
        # trimp of real performance
        trimp[i] = data.loc[data['date'] == date[i]]['trimp']

        # position of trimp
        position[i] = data.loc[data['date'] == date[i]].index.values

    del data['trimp']
    data['trimp'] = zeros
    for i in range(0, len(date)):
        # dodanie tylko wyselekcjonowanych trimp
        data.at[position[i], 'trimp'] = trimp[i]

    time = [5.29, 5.45, 5.30, 5.50, 5.27, 5.27]
    # print(cp(time))
    p0 = cp(time)[-1]

    while (abs(sum(maks) - sum(cp(time))) > 1):
        t1 = random.randint(40, 50)
        t2 = random.randint(10, 20)
        p0 = random.randint(300, 500)

        for i in range(0, len(date)):
            # dodanie tylko wyselekcjonowanych trimp
            maks[i] = banistermodel(data, t1, t2, p0, 1, 2)['performance'].loc[position[i]:position[i] + 10].max()
        #print(abs(sum(maks) - sum(cp(time))))


    return t1, t2, p0


def prediction(amount):
    data = filtration(1, 5)
    end = data['date'].iloc[-1]

    date_time_obj2 = datetime.datetime.strptime(end, '%Y-%m-%d')
    delta = datetime.timedelta(1)
    table = []
    zeros = np.zeros([amount], dtype=float)

    for i in range(0, amount):
        date_time_obj2 = date_time_obj2 + delta
        string = date_time_obj2
        startstring = string.strftime('%Y-%m-%d')
        temp = {'date': startstring, 'trimp': 0}
        data = data.append(temp, ignore_index=True)
    return data




t1 = fit(filtration(1, 5))[0]
t2 = fit(filtration(1, 5))[1]
p0 = fit(filtration(1, 5))[2]



data = filtration(1,5)

print(len(data))
print(len(df['heartRate']))

# Calculation of Training Monotony of last seven days
avg = data['trimp'].iloc[-7:-1].mean()
std = statistics.stdev(data['trimp'].iloc[-7:-1])

if (avg <= 0 and avg>10):
    monotony = "no training bout"

elif (std < 10):
    monotony = "high training monotonny"

elif (std > 0 and avg > 0):
    monotony = avg / std

print(std)
print(avg)

#strain = data['trimp'].iloc[-7:-1].sum() * monotony



predict = banistermodel(prediction(10), t1, t2, p0, 1, 2)

app = dash.Dash(__name__)

app.layout = html.Div(
    children=[
        html.H1(children="Model of Performance",),
        html.P(children="History",),

        dcc.Graph(
            figure={
                "data": [
                    {
                        "x": data["date"],
                        "y": data["trimp"],
                        "type": "bar",

                    },
                ],
                "layout": {"title": "TRIMP collection"},
            },
        ),
        html.P(children=t1,),
        html.P(children=t2,),
        html.P(children=p0,),
        html.P(children="Training Monotony",),
        html.P(children=monotony,),
        dcc.Graph(
            figure={
                "data": [
                    {
                        "x": predict["date"],
                        "y": predict["performance"],
                        "type": "lines",

                    },
                ],
                "layout": {"title": "performance prediction"},
            },
        ),
    ]
)

if __name__ == "__main__":
    app.run_server(debug=True)


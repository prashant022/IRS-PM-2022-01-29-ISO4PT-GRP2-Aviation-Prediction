from flask import Flask, request, render_template
from flask_cors import cross_origin
import pickle
import pandas as pd
import aviation_cf_rating as cf

app = Flask(__name__)
model = pickle.load(open("aviation_rf.pkl", "rb"))


@app.route("/")
@cross_origin()
def home():
    return render_template("home.html")

@app.route("/predict", methods = ["GET", "POST"])
@cross_origin()
def predict():
    if request.method == "POST":

        # Traveller name
        traveller = request.form["traveller"]

        # Date_of_Journey
        date_dep = request.form["Dep_Time"]
        Journey_day = int(pd.to_datetime(date_dep, format="%Y-%m-%dT%H:%M").day)
        Journey_month = int(pd.to_datetime(date_dep, format ="%Y-%m-%dT%H:%M").month)
        # print("Journey Date : ",Journey_day, Journey_month)

        # Departure
        Dep_hour = int(pd.to_datetime(date_dep, format ="%Y-%m-%dT%H:%M").hour)
        Dep_min = int(pd.to_datetime(date_dep, format ="%Y-%m-%dT%H:%M").minute)
        # print("Departure : ",Dep_hour, Dep_min)

        # Arrival
        date_arr = request.form["Arrival_Time"]
        Arrival_hour = int(pd.to_datetime(date_arr, format ="%Y-%m-%dT%H:%M").hour)
        Arrival_min = int(pd.to_datetime(date_arr, format ="%Y-%m-%dT%H:%M").minute)
        # print("Arrival : ", Arrival_hour, Arrival_min)

        # Duration
        dur_hour = abs(Arrival_hour - Dep_hour)
        dur_min = abs(Arrival_min - Dep_min)
        # print("Duration : ", dur_hour, dur_min)

        # Total Stops
        Total_stops = int(request.form["stops"])
        # print(Total_stops)

        # Airline
        # AIR ASIA = 0 (not in column)
        airline=request.form['airline']

        if(airline=='Silk Air'):
            Silk_Air = 1
            Scoot = 0
            Jetstar_Asia = 0
            Multiple_carriers = 0
            ScootBiz = 0
            Singapore_Airlines = 0
            Malaysia_Airlines = 0
            Multiple_carriers_Premium_economy = 0
            Silk_Air_Business = 0
            Singapore_Airlines_Premium_Economy = 0
            Silk_Air_Premium_Economy = 0 

        elif (airline=='Scoot'):
            Silk_Air = 0
            Scoot = 1
            Jetstar_Asia = 0
            Multiple_carriers = 0
            ScootBiz = 0
            Singapore_Airlines = 0
            Malaysia_Airlines = 0
            Multiple_carriers_Premium_economy = 0
            Silk_Air_Business = 0
            Singapore_Airlines_Premium_Economy = 0
            Silk_Air_Premium_Economy = 0 

        elif (airline=='Jetstar Asia'):
            Silk_Air = 0
            Scoot = 0
            Jetstar_Asia = 1
            Multiple_carriers = 0
            ScootBiz = 0
            Singapore_Airlines = 0
            Malaysia_Airlines = 0
            Multiple_carriers_Premium_economy = 0
            Silk_Air_Business = 0
            Singapore_Airlines_Premium_Economy = 0
            Silk_Air_Premium_Economy = 0 
            
        elif (airline=='Multiple carriers'):
            Silk_Air = 0
            Scoot = 0
            Jetstar_Asia = 0
            Multiple_carriers = 1
            ScootBiz = 0
            Singapore_Airlines = 0
            Malaysia_Airlines = 0
            Multiple_carriers_Premium_economy = 0
            Silk_Air_Business = 0
            Singapore_Airlines_Premium_Economy = 0
            Silk_Air_Premium_Economy = 0 
            
        elif (airline=='ScootBiz'):
            Silk_Air = 0
            Scoot = 0
            Jetstar_Asia = 0
            Multiple_carriers = 0
            ScootBiz = 1
            Singapore_Airlines = 0
            Malaysia_Airlines = 0
            Multiple_carriers_Premium_economy = 0
            Silk_Air_Business = 0
            Singapore_Airlines_Premium_Economy = 0
            Silk_Air_Premium_Economy = 0 
            
        elif (airline=='Singapore Airlines'):
            Silk_Air = 0
            Scoot = 0
            Jetstar_Asia = 0
            Multiple_carriers = 0
            ScootBiz = 0
            Singapore_Airlines = 1
            Malaysia_Airlines = 0
            Multiple_carriers_Premium_economy = 0
            Silk_Air_Business = 0
            Singapore_Airlines_Premium_Economy = 0
            Silk_Air_Premium_Economy = 0

        elif (airline=='Malaysia Airlines'):
            Silk_Air = 0
            Scoot = 0
            Jetstar_Asia = 0
            Multiple_carriers = 0
            ScootBiz = 0
            Singapore_Airlines = 0
            Malaysia_Airlines = 1
            Multiple_carriers_Premium_economy = 0
            Silk_Air_Business = 0
            Singapore_Airlines_Premium_Economy = 0
            Silk_Air_Premium_Economy = 0

        elif (airline=='Multiple carriers Premium economy'):
            Silk_Air = 0
            Scoot = 0
            Jetstar_Asia = 0
            Multiple_carriers = 0
            ScootBiz = 0
            Singapore_Airlines = 0
            Malaysia_Airlines = 0
            Multiple_carriers_Premium_economy = 1
            Silk_Air_Business = 0
            Singapore_Airlines_Premium_Economy = 0
            Silk_Air_Premium_Economy = 0

        elif (airline=='Silk Air Business'):
            Silk_Air = 0
            Scoot = 0
            Jetstar_Asia = 0
            Multiple_carriers = 0
            ScootBiz = 0
            Singapore_Airlines = 0
            Malaysia_Airlines = 0
            Multiple_carriers_Premium_economy = 0
            Silk_Air_Business = 1
            Singapore_Airlines_Premium_Economy = 0
            Silk_Air_Premium_Economy = 0

        elif (airline=='Singapore Airlines Premium economy'):
            Silk_Air = 0
            Scoot = 0
            Jetstar_Asia = 0
            Multiple_carriers = 0
            ScootBiz = 0
            Singapore_Airlines = 0
            Malaysia_Airlines = 0
            Multiple_carriers_Premium_economy = 0
            Silk_Air_Business = 0
            Singapore_Airlines_Premium_Economy = 1
            Silk_Air_Premium_Economy = 0
            
        elif (airline=='Silk Air Premium Economy'):
            Silk_Air = 0
            Scoot = 0
            Jetstar_Asia = 0
            Multiple_carriers = 0
            ScootBiz = 0
            Singapore_Airlines = 0
            Malaysia_Airlines = 0
            Multiple_carriers_Premium_economy = 0
            Silk_Air_Business = 0
            Singapore_Airlines_Premium_Economy = 0
            Silk_Air_Premium_Economy = 1

        else:
            Silk_Air = 0
            Scoot = 0
            Jetstar_Asia = 0
            Multiple_carriers = 0
            ScootBiz = 0
            Singapore_Airlines = 0
            Malaysia_Airlines = 0
            Multiple_carriers_Premium_economy = 0
            Silk_Air_Business = 0
            Singapore_Airlines_Premium_Economy = 0
            Silk_Air_Premium_Economy = 0

        Source = request.form["Source"]
        if (Source == 'Singapore'):
            s_Singapore = 1
            s_Bangkok = 0
            s_Kuala_Lumpur = 0
            s_Hanoi = 0

        elif (Source == 'Bangkok'):
            s_Singapore = 0
            s_Bangkok = 1
            s_Kuala_Lumpur = 0
            s_Hanoi = 0

        elif (Source == 'Kuala_Lumpur'):
            s_Singapore = 0
            s_Bangkok = 0
            s_Kuala_Lumpur = 1
            s_Hanoi = 0

        elif (Source == 'Hanoi'):
            s_Singapore = 0
            s_Bangkok = 0
            s_Kuala_Lumpur = 0
            s_Hanoi = 1

        else:
            s_Singapore = 0
            s_Bangkok = 0
            s_Kuala_Lumpur = 0
            s_Hanoi = 0

        # Destination

        Destination = request.form["Destination"]
        if (Destination == 'Bali_Denpasar'):
            d_Bali_Denpasar = 1
            d_Singapore = 0
            d_Phuket = 0
            d_Bangkok = 0
        
        elif (Destination == 'Singapore'):
            d_Bali_Denpasar = 0
            d_Singapore = 1
            d_Phuket = 0
            d_Bangkok = 0

        elif (Destination == 'Phuket'):
            d_Bali_Denpasar = 0
            d_Singapore = 0
            d_Phuket = 1
            d_Bangkok = 0

        elif (Destination == 'Bangkok'):
            d_Bali_Denpasar = 0
            d_Singapore = 0
            d_Phuket = 0
            d_Bangkok = 1

        else:
            d_Bali_Denpasar = 0
            d_Singapore = 0
            d_Phuket = 0
            d_Bangkok = 0

        prediction=model.predict([[
            Total_stops,
            Journey_day,
            Journey_month,
            Dep_hour,
            Dep_min,
            Arrival_hour,
            Arrival_min,
            dur_hour,
            dur_min,
            Jetstar_Asia,
            Malaysia_Airlines,
            Scoot,
            Silk_Air,
            Silk_Air_Business,
            Multiple_carriers,
            Multiple_carriers_Premium_economy,
            ScootBiz,
            Silk_Air_Premium_Economy,
            Singapore_Airlines,
            Singapore_Airlines_Premium_Economy,
            s_Hanoi,
            s_Singapore,
            s_Bangkok,
            s_Kuala_Lumpur,
            d_Bali_Denpasar,
            d_Singapore,
            d_Phuket,
            d_Bangkok,
            d_Singapore
        ]])


        output=round(prediction[0],2)

        rating_text = cf.get_airline_ratings(airline,traveller)

        return render_template('home.html',prediction_text="Your Flight price is SGD. {} ".format(output),rating_text=rating_text)


    return render_template("home.html")


if __name__ == "__main__":
    app.run(debug=True)

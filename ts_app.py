import streamlit as st
import pandas as pd
from joblib import load
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go



#from sklearn.metrics import mean_absolute_error

def app():
    
    with open('data\PJME_hourly.csv') as f:
        data = pd.read_csv(f)
        
    data = data.rename(columns={'Datetime':'Time', 'PJME_MW':'Demand'})
    data['Time'] = pd.to_datetime(data['Time'], format='%Y-%m-%d %H:%M:%S')
    data = data.set_index('Time')
    data = data.loc[~data.index.duplicated(), :]
    data = data.asfreq('60min')
    data = data.fillna(method = 'bfill')
    data = data.loc['2015-01-01 00:00:00': '2017-12-31 23:00:00']
    end_train = '2017-11-30 23:59:00'
    data_train = data.loc[: end_train, :]
    data_test  = data.loc[end_train:, :]
    
       
    page = st.sidebar.selectbox("", ['Homepage', 'Electricity demand',
                                                  'Demand distribution','Daily Forecast'])
    
    

    if page == 'Homepage':
        image = Image.open('images\main.jpg')
        
        st.image(image)
        st.title('Time Series Forecasting with [Skforecast](https://joaquinamatrodrigo.github.io/skforecast/0.4.3/index.html)')
        st.markdown(""" ### We build a time series forecasting model to predict the hourly energy demand\
                    for a certain region in the Eastern United States and use this app to showcase our work. The time series data we use is \
                    [here](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption). """)
        st.markdown(""" ### This app can be used to:""")                 
                 
        st.markdown(""" 
                  * get the interactive plot of the time series by choosing **Electricity demand** from the dropdown menu on the sidebar. 
                  * get the interactive plot of *Monthly*, *Weekly* and *Daily* demand distribution by choosing 
                  **Demand distribution** from the dropdown menu on the sidebar. 
                  * get the daily forecast for energy demand using a pretrained model on a test data
                  by choosing **Daily forecast** from the dropdown menu on the sidebar.""") 
        
                                        
        
    elif page == 'Electricity demand':
        
              
           
        fig = go.Figure()
        fig.add_scatter(x=data_train.index, y=data_train.Demand, name = 'Train')
        fig.add_scatter(x=data_test.index, y=data_test.Demand, name ='Test',  line={'color': 'orange'})
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""This is the plot of the time series between *2015/01/01 - 2017/12/31*. 
                    The whole time series is actually between *2002/12/31 - 2017/12/31* recorded at hourly intervals. 
                    The plot shows that the electricity demand has annual seasonality with large demand peaks during the
                    summer months and smaller demand peaks during the winter months. The intraday pattern of the time series 
                    can be explored by expanding the plot and zooming in on the time series which shows a weekly seasonality
                    with higher consumption during the work week (Monday to Friday) and lower consumption on weekends.""")
        st.markdown("""Finally, we have divided our time series into training and test set for the purpose of 
                    optimizing the hyperparameters of the model and evaluating its predictive capability. """)            
        
           
        
      
           
           
    elif page == 'Demand distribution':         
             
       dist = st.sidebar.selectbox("Boxplot demand distribution grouped by",['Monthly','Weekly', 'Daily'])
       
       if dist == 'Monthly':      
                                 
           data['month'] = data.index.month
           fig = px.box(data, x=data['month'],y ='Demand')
           st.plotly_chart(fig, use_container_width=True)
           
           st.markdown("""It is observed that there is an annual seasonality, with higher (median) demand values 
                       in July and August, and smaller (median) demand values in January, February.""")
           
           
       if dist == 'Weekly':       
                    
    
           data['week_day'] = data.index.day_of_week + 1                    
           fig = px.box(data, x=data['week_day'],y ='Demand')
           st.plotly_chart(fig, use_container_width=True)
           
           st.markdown("""Weekly seasonality shows lower demand values during the weekend.""")
           
           
       if dist == 'Daily':
           
           
           data['hour_day'] = data.index.hour + 1
           fig = px.box(data, x=data['hour_day'],y ='Demand')
           st.plotly_chart(fig, use_container_width=True)
           
           st.markdown("""There is also a daily seasonality, with demand decreasing substantially 
                       after midnight and increasing after 6am with a peak at 7pm .""")
              
        
    else:
        forecaster = load('model/forecaster.py')
        predictions = forecaster.predict(744) #31days*24 hours which is the number of hours in Dec.
        days = ['Dec ' +str(i) for i in range(1,32)]
        dates = pd.date_range('2017-12-01', '2017-12-31')
        dates_dict = dict(zip(days, dates))
        prediction_day = st.sidebar.selectbox("Forecasted electricity demand for",days)
        forecast_date = dates_dict[str(prediction_day)].strftime('%Y-%m-%d')
        forecast = predictions[forecast_date:forecast_date]
        
        st.markdown(""" Here we use a model built using `skforecast`'s `ForecasterAutoreg` class which 
                    uses Ridge regression and a time window of 24 lags. This means that the model uses the 
                    previous 24 hours as predictors. The model is trained on the data 
                    between the dates *2015-01-01 00:00:00* and *2017-12-31 23:00:00*. The trained model
                    can be used to generate daily energy forecast for the month of December which is the test data.""")
        
        y_true = forecast
        y_pred = data.loc[forecast.index,'Demand']
                          
        fig = go.Figure()
        fig.add_scatter(x=forecast.index, y=y_pred, name = 'Forecasted Demand')
        fig.add_scatter(x=forecast.index, y=y_true, name ='Actual Demand',  line={'color': 'red'})
        
        st.plotly_chart(fig, use_container_width=True)
        
        #mean_abs_err = mean_absolute_error(y_true, y_pred)
        #st.write(f'Mean absolute error of the forecast for {prediction_day} is: ${mean_abs_err}$') 
   


if __name__ == '__main__':
    app()
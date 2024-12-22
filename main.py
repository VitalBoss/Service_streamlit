import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import requests
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA

def get_season(date):
    m = date.month
    x = m%12 // 3 + 1
    if x == 1:
        return "winter"
    if x == 2:
        return "spring"
    if x == 3:
        return "summer"
    if x == 4:
        return "autumn"
    else:
        None

def prepare_city(df : pd.DataFrame, city : str, window_size : int = 30) -> pd.DataFrame:
        buf = df[df['city'] == city]
        buf[f'smooth_30_mean'] = buf['temperature'].rolling(window_size).mean()
        buf[f'smooth_30_std'] = buf['temperature'].rolling(window_size).std()
        buf['is_anomaly'] = (buf['smooth_30_mean'] - 2*buf['smooth_30_std'] > buf['temperature']) | (buf['smooth_30_mean'] + 2*buf['smooth_30_std'] < buf['temperature'])        
        buf['max_temperature'] = buf['temperature'].max()
        buf['min_temperature'] = buf['temperature'].min()
        buf['mean_temperature'] = buf['temperature'].mean()
        buf['std_temperature'] = buf['temperature'].std()
        return buf

def get_forecast(df : pd.DataFrame, city : str, days : int):
    y = np.array(df[df['city'] == city]['temperature'])
    model = ARIMA(y, order=(2, 3, 3)) 
    results = model.fit()
    pred = results.forecast(steps=days)
    return pred

def get_profile_season(data : pd.DataFrame) -> dict:
        def to_integer(dt_time):
            return 10000*dt_time.year + 100*dt_time.month + dt_time.day
        df = data.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        group = df[['timestamp', 'temperature',	'season']].groupby(['season', 'timestamp']).mean()
        group['season'] = [elem[0] for elem in group.index]
        group['timestamp'] = [elem[1] for elem in group.index]
        group = group.reset_index(drop=True)
        model = LinearRegression()
        dt_season = dict()

        for season in group['season'].unique():
            df_season = group[group['season'] == season].copy()

            X = pd.to_datetime(df_season[df_season['season'] == season]['timestamp']).apply(lambda x: to_integer(x))
            model.fit(np.array(X).reshape(-1, 1), df_season[df_season['season'] == season]['temperature'])
            df_season['pred'] = model.predict(np.array(X).reshape(-1, 1))
            df_season['trend_coef'] = model.coef_[0]

            df_season['avg_temp'] = df_season['temperature'].mean()
            df_season['std_temp'] = df_season['temperature'].std()

            dt_season[season] = df_season

        return dt_season


def create_date_list(start_date, days_count):
    dates = []
    for i in range(days_count + 1):
        new_date = start_date + timedelta(days=i)
        dates.append(new_date.strftime("%Y-%m-%d"))
    return dates

df, dt_seasons = None, None

st.title("Анализ погодных данных")

uploaded_file = st.file_uploader("Загрузите файл с историческими данными:", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    st.write("Пожалуйста, загрузите файл")


cities = ['New York', 'London', 'Paris', 'Tokyo', 'Moscow', 'Sydney',
       'Berlin', 'Beijing', 'Rio de Janeiro', 'Dubai', 'Los Angeles',
       'Singapore', 'Mumbai', 'Cairo', 'Mexico City']
selected_city = st.selectbox("Выберите город:", cities)

# API key
api_key = st.text_input("Введите ваш API-ключ OpenWeatherMap:")

if not df is None:
    data = prepare_city(df=df, city=selected_city)
    st.subheader("Описательная статистика")
    st.write(f"Минимальная температура = {round(data['min_temperature'].iloc[0], 1)}")
    st.write(f"Максимальная температура = {round(data['max_temperature'].iloc[0], 1)}")
    st.write(f"Средняя температура = {round(data['mean_temperature'].iloc[0], 1)}")
    st.write(f"Стандартное отклонение температуры = {round(data['std_temperature'].iloc[0], 1)}")

    fig = go.Figure()
    for index, row in data.iterrows():
        if row['is_anomaly']:
            fig.add_trace(go.Scatter(x=[row['timestamp']], y=[row['temperature']], mode='markers', marker=dict(color='red')))
        else:
            fig.add_trace(go.Scatter(x=[row['timestamp']], y=[row['temperature']], mode='markers', marker=dict(color='blue')))

    fig.update_layout(title="Температура со значениями аномалий", xaxis_title="Дата", yaxis_title="Температура")
    st.plotly_chart(fig)


    seasons = ['winter', 'spring', 'summer', 'autumn']
    selected_season= st.selectbox("Выберите сезон:", seasons)
    dt_seasons = get_profile_season(df)
    if dt_seasons:
        data_season = dt_seasons[selected_season]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data_season['timestamp'], y=data_season['temperature'], name="истинная температура"))
        fig.add_trace(go.Scatter(x=data_season['timestamp'], y=data_season['pred'], name="тренд"))
        dt = {
            'winter' : 'зимой',
            'spring' : 'весной',
            'autumn' : 'осенью',
            'summer' : 'летом'
        }
        fig.update_layout(title=f"Профиль температуры {dt[selected_season]} с трендом. Mean = {round(data_season['avg_temp'].iloc[0], 1)}; Std = {round(data_season['std_temp'].iloc[0], 1)}", xaxis_title="Дата", yaxis_title="Температура")
        st.plotly_chart(fig)

        if data_season['trend_coef'].iloc[0] > 0:
            st.write(f"Временной ряд имеет положительный тренд")
        else:
            st.write(f"Временной ряд имеет отрицательный тренд")


response=None
if api_key:
    url = f"https://api.openweathermap.org/data/2.5/weather?q={selected_city}&appid={api_key}"
    response = requests.get(url)
else:
    st.warning("API-ключ не введен. Данные о текущей погоде недоступны.")

if not response is None:
    if response.status_code == 200:
        current_weather = response.json()
        current_temp = current_weather["main"]["temp"] - 273.15  # переводим Кельвины в Цельсии
        st.subheader(f"Текущая температура в {selected_city}: {current_temp:.1f}°C")
        x = get_season(datetime.now())
        print(x)
        mean, std = dt_seasons[x]['avg_temp'].iloc[0], dt_seasons[x]['std_temp'].iloc[0]
        if mean-3*std < current_temp < mean+3*std:
            st.write("Данная температура в пределах нормы для текущего сезона (winter)")
        else:
            st.write("Данная температура не является нормой для текущего сезона (winter)")
    elif response.status_code == 401:
        st.error(r'{"cod":401, "message": "Invalid API key. Please see https://openweathermap.org/faq#error401 for more info."}')


with st.form(key="arima_form"):
    n_steps = st.slider(label="Количество шагов для прогноза:", min_value=1, max_value=10, value=5)
    submit_button = st.form_submit_button(label="Построить прогноз")

def plot_forecast(forecast):
    fig, ax = plt.subplots(figsize=(12, 6))
    last_date = dt_seasons[selected_season]['timestamp'].iloc[-1]
    dates = create_date_list(last_date, n_steps)
    ax.plot(dates[1:], forecast, color='red', marker='o', linestyle='--', label='Forecast')
    ax.legend()
    return fig

if submit_button:
    forecast = get_forecast(df, selected_city, n_steps)
    fig = plot_forecast(forecast)
    st.write(f"График предсказания временного ряда (последняя дата - {dt_seasons[selected_season]['timestamp'].iloc[-1]}):")
    st.pyplot(fig)



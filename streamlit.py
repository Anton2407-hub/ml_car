import joblib
import streamlit as st
import pandas as pd
import numpy as np


@st.cache_data  # 👈 Add the caching decorator
def load_data(url):
    df = pd.read_csv(url)
    return df

model = joblib.load('model/ml_pipeline.pkl')

# Загрузка данных из CSV файла
data = load_data('data/car_data.csv')

# Создание пустой Series для хранения выбранных параметров
selected_features = pd.Series(index = model.feature_names_in_)
selected_features.loc[0] = np.nan

# Первый шаг - выбор бренда
selected_features['brand'] = st.selectbox('Выберите марку автомобиля', data['brand'].unique())

# Фильтрация данных по выбранной марке
filtered_data = data[data['brand'] == selected_features['brand']]

# Второй шаг - выбор модели
selected_features['model'] = st.selectbox('Выберите модель автомобиля', filtered_data['model'].unique())
filtered_data = filtered_data[filtered_data['model'] == selected_features['model']]

# Третий шаг - выбор поколения
selected_features['поколение'] = st.selectbox('Выберите поколение автомобиля', filtered_data['поколение'].unique())
filtered_data = filtered_data[filtered_data['поколение'] == selected_features['поколение']]

# Четвёртый шаг - выбор модификации
selected_features['модификация'] = st.selectbox('Выберите модификацию автомобиля', filtered_data['модификация'].unique())
filtered_data = filtered_data[filtered_data['модификация'] == selected_features['модификация']]

# Пятый шаг - выбор комплектации
selected_features['комплектация'] = st.selectbox('Выберите комплектацию автомобиля', filtered_data['комплектация'].unique())
filtered_data = filtered_data[filtered_data['комплектация'] == selected_features['комплектация']]

# Шестой шаг - выбор типа кузова
selected_features['тип кузова'] = st.selectbox('Выберите тип кузова автомобиля', filtered_data['тип кузова'].unique())
filtered_data = filtered_data[filtered_data['тип кузова'] == selected_features['тип кузова']]

# Седьмой шаг - выбор коробки передач
selected_features['коробка передач'] = st.selectbox('Выберите тип коробки передач', filtered_data['коробка передач'].unique())
filtered_data = filtered_data[filtered_data['коробка передач'] == selected_features['коробка передач']]

# Восьмой шаг - выбор типа двигателя
selected_features['тип двигателя'] = st.selectbox('Выберите тип двигателя', filtered_data['тип двигателя'].unique())
filtered_data = filtered_data[filtered_data['тип двигателя'] == selected_features['тип двигателя']]

# Девятый шаг - выбор привода
selected_features['привод'] = st.selectbox('Выберите тип привода', filtered_data['привод'].unique())
filtered_data = filtered_data[filtered_data['привод'] == selected_features['привод']]

# Десятый шаг - выбор года выпуска
selected_features['год выпуска'] = st.selectbox('Выберите год выпуска автомобиля', filtered_data['год выпуска'].unique())
filtered_data = filtered_data[filtered_data['год выпуска'] == selected_features['год выпуска']]

# Одиннадцатый шаг - выбор пробега
selected_features['пробег'] = st.number_input('Введите текущий пробег автомобиля')

# Двенадцатый шаг - выбор цвета
selected_features['цвет'] = st.selectbox('Выберите цвет автомобиля', filtered_data['цвет'].unique())
filtered_data = data[data['цвет'] == selected_features['цвет']]

# Тринадцатый шаг - выбор типа руля
selected_features['руль'] = st.selectbox('Выберите тип руля', filtered_data['руль'].unique())

# Четырнадцатый шаг - выбор расположения
selected_features['расположение'] = st.selectbox('Выберите расположение автомобиля', filtered_data['расположение'].unique())

# Отображение выбранных параметров
st.write('Выбранные параметры:')
st.write(selected_features)


if st.button('Оценить авто'):
    # Преобразование Series в DataFrame
    selected_vector = selected_features.to_frame().transpose()

    # Предсказание с использованием модели
    prediction = np.round(model.predict(selected_vector))
    
    # Отображение результата
    st.write(f'Прогноз стоимости автомобиля: {prediction} рублей')

import pandas as pd
from sklearn.linear_model import LinearRegression
import streamlit as st
import matplotlib.pyplot as plt

df = pd.read_csv('indonesian_salary_by_region.csv')

st.title("Prediksi Gaji UMP/UMK Indonesia")
st.markdown("---")

region = df['REGION'].unique()

input_region = st.selectbox("Pilih Provinsi", options= region)

df_selected = df[df['REGION'] == input_region].copy()

X = df_selected[['YEAR']] # independent 
y = df_selected['SALARY'] # dependent

model = LinearRegression()
model.fit(X, y)

min_year = df_selected['YEAR'].min()
max_year = df_selected['YEAR'].max()
input_year = st.number_input("Masukan Tahun: ", min_value = min_year,  value = max_year+1)

prediction = model.predict([[input_year]])

st.metric( label=f'Prediksi Gaji di Provinsi {input_region} di Tahun {input_year}',value= prediction[0])

st.subheader("Graph hooh")

fig, ax = plt.subplots()
ax.scatter (X, y, label=f'diem aja bersantai {input_region} 1997-2025', color = 'blue')
ax.plot(X, model.predict(x), color ='red', label="gataulah")

ax.scatter([input_year], [prediction], color='green', label=f'pred thn 26{input_year}')

ax.set_xlabel("Tahun")
ax.set_ylabel("Gaji")
ax.set_title("tulis sendiri males aku")
ax.legend()

st.pyplot(fig)

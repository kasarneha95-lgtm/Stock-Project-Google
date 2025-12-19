
import streamlit as st  # frontend UI design
import numpy as np
import joblib
import yfinance as yf

model = joblib.load('stock_prediction_google.pkl')
st.title('Stock Market Prediction')   # main title
st.write('Enter inputs to predict estimated Stock Closeing price !')  # subtitle
st.header('Stock Market Predictor')

stock =st.text_input('Enter Stock Symbol', 'GOOG')
start =st.date_input('Enter Start Date','2012-01-01')
end = st.date_input('Enter End Date','2025-12-31')

data = yf.download(stock, start ,end)

st.subheader('Stock Data')
st.write(data)

#open_price = st.number_input('Stock Open Price',step=10.0)
#high_price = st.number_input('Stock High Price',step=10.0)
#low_price = st.number_input('Stock Low Price',step=10.0)
#volume = st.number_input('Stock Volume',step=50.0)


if st.button('Perdict Price'):
  X = np.array([[stock,start,end]])
  pred = model.predict(X)[0]
  st.success(f'Estimated Stock Data : {pred:.2f}')

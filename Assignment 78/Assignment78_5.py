import streamlit as st

st.title("Celsius to Farenheit Converter")

celsius = st.number_input("Enter temperature in celsius")

if st.button("Convert"):
    fahrenheit = (celsius * 9 / 5) + 32
    st.success(f"Temperature in fahrenheit = {fahrenheit:.2f} 'F")
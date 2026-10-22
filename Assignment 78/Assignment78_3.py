import streamlit as st

st.title("Simple Intrest Calculator")

principle = st.number_input("Enter your amount : ",step = 1000,value=0)
rateofinterest = st.number_input("Enter interest rate : ",step = 0.1,value=0.0)
time = st.number_input("Enter time duration : ",value=0.0)

if st.button("Calculate"):
    SimpleInterest = (principle * rateofinterest * time) / 100
    st.success(f"Simple Interest is : {SimpleInterest:.2f}")
import streamlit as st

st.title("Addition of two numbers")

st.header("Addition Calculator")

no1 = st.number_input("Enter first number",value=0,step=1)
no2 = st.number_input("Enter second number",value=0,step=1)

if st.button("Add"):
    result = no1 + no2

    st.success("Addition is : "+str(result))
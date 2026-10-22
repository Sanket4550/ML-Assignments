import streamlit as st

st.title("Even or Odd Number")

num = st.number_input("Enter your number : ",step = 1,value=0)

if st.button("Check"):
    if int(num) % 2 == 0:
        st.success("The number " + str(num) + " is even")
    else:
        st.success("The number "+ str(num) + " is odd")
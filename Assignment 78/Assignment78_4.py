import streamlit as st

st.title("Student Result Calculator")

name = st.text_input("Enter your Name : ")
math = st.number_input("Enter marks scored in math : ",step = 1,max_value=100)
english = st.number_input("Enter marks scored in english : ",step = 1,max_value=100)
marathi = st.number_input("Enter marks scored in marathi : ",step = 1,max_value=100)


if st.button("Calculate"):
    total = math + english + marathi
    percentage = total / 3


    st.write("Student name : ",name)
    st.write("Total marks : ",total," / 300")
    st.write(f"Percentage : {percentage:.2f} %")

    if math >= 40 and english >= 40 and marathi >= 40:
        st.success("Student is Passed")
    else:
        st.success("Student is Failed")
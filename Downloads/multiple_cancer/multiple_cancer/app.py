import streamlit as st
from auth.auth_utils import create_users_table, add_user, login_user
from cancers import breast_cancer, lung_cancer, liver_cancer, skin_cancer, coletrol_cancer
from features import hospital_map, diet_yoga, chatbot

def main():
    st.set_page_config(page_title="Cancer Prediction App", layout="wide")
    menu = ["Login", "Sign Up"]
    choice = st.sidebar.selectbox("Menu", menu)

    create_users_table()  # make sure table exists

    if choice == "Login":
        st.subheader("Login to your account")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            result = login_user(username, password)
            if result:
                st.success(f"Welcome {username} 👋")
                app_dashboard()
            else:
                st.error("Invalid Username or Password")

    elif choice == "Sign Up":
        st.subheader("Create a new account")
        new_user = st.text_input("Username")
        new_password = st.text_input("Password", type="password")
        if st.button("Sign Up"):
            add_user(new_user, new_password)
            st.success("Account created successfully!")
            st.info("Go to Login Menu to login.")

def app_dashboard():
    st.sidebar.title("Navigation")
    section = st.sidebar.radio("Go to", 
                ["Breast Cancer", "Lung Cancer", "Liver Cancer", "Skin Cancer", "Coletrol Cancer",
                 "Hospital Map", "Diet Plan", "Yoga Plan", "AI Chatbot"])

    if section == "Breast Cancer":
        breast_cancer.predict_breast()
    elif section == "Lung Cancer":
        lung_cancer.predict_lung()
    elif section == "Liver Cancer":
        liver_cancer.predict_liver()
    elif section == "Skin Cancer":
        skin_cancer.predict_skin()
    elif section == "Coletrol Cancer":
        coletrol_cancer.predict_coletrol()
    elif section == "Hospital Map":
        hospital_map.show_map()
    elif section == "Diet Plan":
        diet_plan.show_diet()
    elif section == "Yoga Plan":
        yoga_plan.show_yoga()
    elif section == "AI Chatbot":
        chatbot.chat_ui()

if __name__ == '__main__':
    main()


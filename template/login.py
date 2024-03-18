import streamlit as st
import smtplib
import pandas as pd
import pdf2image
import tempfile
import subprocess
from subprocess import Popen
import random
import pandas as pd

# Set the background color using CSS
st.markdown(
    """
    <style>
    body {
        background-color: #f0f0f0;
    }
    </style>
    """, unsafe_allow_html=True
)

# Function to add a background image using CSS
def set_background_image(image_url):
    background_style = f"""
        <style>
        body {{
            background-image: url('{image_url}');
            background-size: cover;
        }}
        </style>
    """
    st.markdown(background_style, unsafe_allow_html=True)

# Set the background image for the "Login" section
set_background_image("images/undraw_file_sync_ot38.svg")

# Define the CSV file name to store user data
CSV_FILE = "user_data.csv"

# Function to initialize the CSV file with a header if it doesn't exist
def initialize_csv():
    try:
        df = pd.read_csv(CSV_FILE)
    except FileNotFoundError:
        df = pd.DataFrame(columns=["Username", "Password"])
        df.to_csv(CSV_FILE, index=False)

# Function to check if a user is an admin
def is_admin(username, password):
    return username == "aniruddha" and password == "Aniruddha@1204"

# Function to check if a user exists in the CSV
def user_exists(username, password):
    df = pd.read_csv(CSV_FILE)
    return any((df["Username"] == username) & (df["Password"] == password))

# Function to send feedback to your email
def send_feedback_to_email(feedback_message):
    # Replace with your App Password and Gmail account
    app_password = "your_generated_app_password"
    sender_email = "your_email@gmail.com"
    receiver_email = "aniruddhakasar2001@gmail.com"

    try:
        # Connect to the SMTP server
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, app_password)

        # Send the email
        server.sendmail(sender_email, receiver_email, feedback_message)
        server.quit()
    except Exception as e:
        st.error(f"Failed to send feedback email: {str(e)}")

# Sidebar for navigation
choice = st.sidebar.radio("Navigation", ["Home", "Login", "Register"])

# Home page
if choice == "Home":
    # Define a dictionary for service information
    services = {
        "Service 1": "Description of Service 1",
        "Service 2": "Description of Service 2",
        "Service 3": "Description of Service 3",
    }

    # Define information about yourself
    about_me = """
    I am Aniruddha Kasar, a Junior Data Scientist with experience in working on various projects. I am passionate about data science and continuously working on innovative solutions.

    This entire project is developed by me, and it serves as a showcase of my skills and achievements.

    You can explore research papers and projects by selecting the tabs from the dropdown menu.
    """

    # Display a dropdown menu for selecting the section
    section = st.selectbox("MENU", ["Select a Section", "Services", "About Us", "Support"])

    # Display the selected section content
    if section == "Services":
        st.header("Services")
        for service, description in services.items():
            st.subheader(service)
            st.write(description)

    elif section == "About Us":
        st.header("About Us")
        st.write(about_me)
        # Display a window for research papers
        st.header("Research Papers")
        st.write("Explore research papers in this window.")
        research_papers = [
            {"title": "IJSREM Manuscript Template",
             "file_path": "F:\stock price prediction\documents\IJSREM-Manuscript-Template.pdf"},
            # Add more research papers if needed
        ]

        for paper in research_papers:
            if st.button(f"View {paper['title']}"):
                with open(paper['file_path'], 'rb') as pdf_file:
                    pdf_file_bytes = pdf_file.read()
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(pdf_file_bytes)
                    st.image(pdf2image.convert_from_path(tmp_file.name))

                # Display certificates
                st.header("Certificates")
                certificates = [
                            {"title": "Certificate 1", "file_path": "certificates/certificate1.pdf"},
                            {"title": "Certificate 2", "file_path": "certificates/certificate2.pdf"},
                            ]

                for certificate in certificates:
                    st.markdown(f"[{certificate['title']}]({certificate['file_path']})")




    elif section == "Support":
        st.header("Support")
        st.write("If you need any assistance, please contact us.")

        # Add your email and mobile number
        st.write("Email: aniruddhakasar2001@gmail.com")
        st.write("Mobile: (+91) 9359205639")

        # Feedback form
        st.subheader("Feedback Form")
        form = st.form(key='feedback-form')

        email = form.text_input("Your Email (Mandatory)")
        query = form.text_input("Query (Mandatory)")
        description = form.text_area("Description (Optional)")

        if form.form_submit_button("Submit Feedback"):
            if not email or not query:
                st.error("Both Email and Query are mandatory fields. Please fill them.")
            else:
                # Process the feedback and send it to your email
                feedback_message = f"Email: {email}\nQuery: {query}\nDescription: {description}"

                # Send feedback to your email address
                send_feedback_to_email(feedback_message)

                st.success("Thank you for your feedback! We will get back to you soon.")


# Login page
elif choice == "Login":
    st.markdown(
        """
        <style>
        .contents {
            text-align: center;
        }
        img {
            max-width: 100%;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<h1 style='text-align: center;'>Sign In to <strong>GlobalStock</strong></h1>", unsafe_allow_html=True)
    st.write("")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Log In"):
        if is_admin(username, password):
            st.success("Logged in as admin.")
            try:
                df = pd.read_csv(CSV_FILE)
            except FileNotFoundError:
                st.error(f"Error: CSV file '{CSV_FILE}' not found.")
                st.stop()  # Stop execution if the file is not found

            # Replace with your logic for running 'global.py' if needed.
        elif user_exists(username, password):
            st.success(f"Logged in as:{username}")
            try:
                result = subprocess.run(["streamlit", "run", "global.py"], capture_output=True, text=True, check=True)
                st.code(result.stdout, language="python")
            except subprocess.CalledProcessError as e:
                st.error(f"Error running the script: {e}")

elif choice == "Register":
    st.markdown("<h1 style='text-align: center;'>Register In to <strong>GlobalStock</strong></h1>", unsafe_allow_html=True)
    st.write("")
    user_data_df = pd.read_csv("user_data.csv")
    new_username = st.text_input("New Username")
    new_password = st.text_input("New Password", type="password")

    if st.button("Register"):
        if new_username in user_data_df["Username"].values:
            st.warning("Username already exists. Please choose another.")
        else:
            is_admin = False
            if new_username == "admin":  # You can customize the admin username
                is_admin = True
            new_user_data = pd.DataFrame({"Username": [new_username], "Password": [new_password], "IsAdmin": [is_admin]})
            user_data_df = pd.concat([user_data_df, new_user_data], ignore_index=True)
            user_data_df.to_csv("user_data.csv", index=False)
            st.success("Registration successful. You can now log in.")
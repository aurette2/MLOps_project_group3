import json
import streamlit as st
import requests
from PIL import Image
from io import BytesIO

import jwt
import time
from datetime import datetime, timezone
from dotenv import load_dotenv
from streamlit_javascript import st_javascript


load_dotenv()
SECRET_KEY= "your_secret_key"
ALGORITHM = "HS256"

# Initialize the screen state using session state
if "screenstate" not in st.session_state:
    st.session_state.screenstate = {
        "login_page": True,
        "logout": False,
        "generate_reports": False,
    }


# Function to authenticate user and get token
def authenticate(username, password):
    try:
        response = requests.post(
            "http://127.0.0.1:8000/token/",
            data={"username": username, "password": password}
        )
        if response.status_code == 200:
            return response.json().get("access_token")
        else:
            st.error("Incorrect username or password.")
            return None
    except Exception as e:
        st.error(f"An error occurred during authentication: {str(e)}")
        return None
 
 
# ____________________________________________________________________________________________________________________________


# Define functions to interact with localStorage
def local_storage_get(key):
    return st_javascript(f"localStorage.getItem('{key}');")

def local_storage_set(key, value):
    value = json.dumps(value)
    return st_javascript(f"localStorage.setItem('{key}', JSON.stringify({value}));")

def local_storage_remove(key):
    return st_javascript(f"localStorage.removeItem('{key}');")

def is_token_valid(token: str):
    try:
        # Decode the token without verifying the signature (just to extract the exp)
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        
        # Get the expiration time (exp) from the payload
        exp_timestamp = payload.get("exp")
        
        # Check if the token has expired
        if exp_timestamp:
            exp_datetime = datetime.fromtimestamp(exp_timestamp, tz=timezone.utc)
            if exp_datetime > datetime.now(timezone.utc):
                return True  # Token is still valid
            else:
                st.error("Token has expired.")
                return False  # Token has expired
        else:
            st.error("No expiration field in token.")
            return False  # No expiration info in token

    except jwt.ExpiredSignatureError:
        st.error("Token has expired.")
        return False  # The token has expired

    except jwt.InvalidTokenError:
        st.error("Invalid token.")
        return False  # Invalid token

# Function to show success message temporarily
def show_temporary_success_message(message, duration=3):
    placeholder = st.empty()  # Create a placeholder for the success message
    placeholder.success(message)  # Display the success message
    time.sleep(duration)  # Wait for 3 seconds
    placeholder.empty()
    
# Function to check if token exists and wait for value
def wait_for_token():
    token = local_storage_get("token")
    print("Hello token", token)
    if token:
        # And it still valid.
        if is_token_valid(token):
            st.session_state.token = token

# Ensure session state is initialized
if "token" not in st.session_state:
    st.session_state.token = None
    # Trigger token fetch from localStorage

    wait_for_token()

# Check if the token exists after fetching
if st.session_state.token:
    # User is authenticated, proceed with other logic
    show_temporary_success_message(f"You are authenticated.")


# Login form
if st.session_state.screenstate["login_page"]:
    # Authentication form
    with st.form("login_form"):
        st.write("Please log in to continue.")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        login_button = st.form_submit_button("Login")

    # Simulate login and store token in localStorage
    if login_button:
        # Simulate token generation after login
        token = authenticate(username, password)
        
    
        if token:
            local_storage_set("token", token)  # Store token in localStorage
            st.session_state.token = token  # Store the token in session state
            st.session_state.screenstate["login_page"] = False  # Hide login page
            st.session_state.screenstate["generate_reports"] = True  # Show report generator
            st.session_state.screenstate["logout"] = True  # Show Logout boutton
            st.success(f"Logged in successfully! Token: {token}")
            
            # Rerun the app to apply the state change
            st.rerun()



# ____________________________________________________________________________________________________________________________    

# Streamlit App
st.title("Radiology Report Generator")



# Show the report generation page if logged in
if st.session_state.screenstate["generate_reports"]:
    st.write("You are now logged in and can generate reports.")
    # File uploader for the chest X-ray image
    uploaded_file = st.file_uploader("Choose a chest X-ray image...", type=["jpg", "jpeg", "png"])

    # Text input for the clinical indication
    indication = st.text_input("Clinical Indication", "Patient presenting with persistent cough, fever, and difficulty breathing. Evaluate for pneumonia.")

    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded Chest X-ray.', use_column_width=True)
        if st.button("Generate Report"):
            with st.spinner('Generating report...'):
                try:
                    img_str = uploaded_file.read()
                    response = requests.post(
                        "http://127.0.0.1:8000/generate_report/",
                        headers={"Authorization": f"Bearer {st.session_state.token}"},
                        files={"file": ("image.jpg", img_str, uploaded_file.type)},
                        data={"indication": indication}
                    )
                    if response.status_code == 200:
                        data = response.json()
                        report = data.get("report", "No report generated.")
                        st.success("Report generated successfully!")
                        st.write(f"**Report:** {report}")
                    else:
                        st.error(f"Error generating the report. Status code: {response.status_code}")
                        st.write(response.text)  # Print the error details
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")


# Logout button to clear token
if st.session_state.screenstate["logout"]:
    if st.button("Logout"):
        local_storage_remove("token")
        st.session_state.token = None

        st.session_state.screenstate["login_page"] = True  # Hide login page
        st.session_state.screenstate["generate_reports"] = False  # Show report generator
        st.session_state.screenstate["logout"] = False  # Show Logout boutton
        st.rerun()


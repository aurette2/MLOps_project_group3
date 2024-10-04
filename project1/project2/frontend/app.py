import streamlit as st
import requests
import streamlit.components.v1 as components
from PIL import Image
import os

# Base URL of FastAPI backend
BASE_URL = "http://localhost:8000"  # Update with your backend URL

# Streamlit App
st.set_page_config(page_title="Medical Image Segmentation", layout="wide")

# Initialize session state for authentication and login status
if 'access_token' not in st.session_state:
    st.session_state.access_token = None

if 'is_logged_in' not in st.session_state:
    st.session_state.is_logged_in = False

# Authentication function
def login(username, password):
    try:
        response = requests.post(f"{BASE_URL}/token", data={"username": username, "password": password})
        if response.status_code == 200:
            st.session_state.access_token = response.json().get('access_token')
            st.session_state.username = username
            st.session_state.is_logged_in = True
            st.success(f"Welcome, {username}!")
        else:
            st.error("Invalid username or password")
    except Exception as e:
        st.error(f"Error logging in: {e}")

# Logout function
def logout():
    st.session_state.access_token = None
    st.session_state.is_logged_in = False
    st.success("Logged out successfully!")

# Check if user is authenticated
def is_authenticated():
    return st.session_state.access_token is not None

# Function to show drift (placeholder)
def show_drift():
    headers = {"Authorization": f"Bearer {st.session_state.access_token}"}
    response = requests.get(f"{BASE_URL}/showdrift/", headers=headers)
    if response.status_code == 200:
        st.success(response.json().get("message"))
    else:
        st.error("Error in fetching drift status.")

# Function to send authenticated request
def authenticated_request(endpoint, method="GET", params=None, json=None):
    headers = {"Authorization": f"Bearer {st.session_state.access_token}"}
    if method == "GET":
        response = requests.get(f"{BASE_URL}{endpoint}", headers=headers, params=params)
    elif method == "POST":
        response = requests.post(f"{BASE_URL}{endpoint}", headers=headers, json=json)
    return response

# ---- MAIN APP LOGIC ----

# Only show login page if the user isn't authenticated yet
if not is_authenticated() and not st.session_state.is_logged_in:
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        login(username, password)
        # Rerun the app to apply the state change
        st.rerun()

# If authenticated, show navigation and operations
if is_authenticated() and st.session_state.is_logged_in:
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Welcome", "Segmentation Prediction", "Model Evaluation", "Drift Detection", "Logout"])
    # # Sidebar for navigation
    # st.sidebar.title("Navigation")
    # page = st.sidebar.selectbox(
    #     "Select a page:",
    #     ["Welcome", "Segmentation Prediction", "Model Evaluation", "Drift Detection", "Logout"]
    # )

    # Welcome Page
    if page == "Welcome":
        st.title(f"Welcome, {st.session_state.username}!")
        st.subheader("Available Operations")
        st.write("- **Segmentation Prediction**: Upload a medical image and predict segmentation.")
        st.write("- **Model Evaluation**: Evaluate the model on the test dataset.")
        st.write("- **Drift Detection**: Check for any data drift.")
        st.write("- **Logout**: End your session.")

    # Segmentation Prediction Page
    if page == "Segmentation Prediction":
        st.title("Segmentation Prediction")
        st.write("Upload a medical image case and make segmentation predictions.")

        # Upload image case
        uploaded_file = st.file_uploader("Choose a case image", type=["png", "jpg", "jpeg", "nii"])
        case_id = st.text_input("Enter Case ID (numeric)", "0")

        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            # Save the uploaded image locally
            case_path = os.path.join("data", uploaded_file.name)
            with open(case_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Prediction
            if st.button("Predict Segmentation"):
                response = authenticated_request(f"/predict/", method="POST", json={"case_path": case_path, "case": case_id})
                if response.status_code == 200:
                    st.success("Prediction successful")
                    st.write(response.json()["prediction"])

                    # Display Predictions by Case ID
                    st.subheader("View Predictions by Case ID")
                    try:
                        numcase = int(st.text_input("Enter the Case ID (integer) for prediction results", case_id))
                        start_slice = st.slider("Select Start Slice", min_value=0, max_value=100, value=60)
                        if st.button("Show Predictions by ID"):
                            show_predicts_response = authenticated_request(f"/showPredictsByID/", method="GET", params={"numcase": numcase, "start_slice": start_slice})
                            if show_predicts_response.status_code == 200:
                                st.success(f"Predictions by Case ID {numcase} displayed")
                            else:
                                st.error("Failed to show predictions by ID")
                    except ValueError:
                        st.error("Please enter a valid integer for Case ID")

                    # Display Segmented Predictions
                    st.subheader("View Predicted Segmentations")
                    samples_list_input = st.text_area("Enter list of samples (comma-separated values)")
                    try:
                        samples_list = [int(item.strip()) for item in samples_list_input.split(',') if item.strip().isdigit()]
                        slice_to_plot = st.slider("Select Slice to Plot", min_value=0, max_value=100, value=50)
                        if st.button("Show Predicted Segmentations"):
                            show_segmented_response = authenticated_request(f"/showPredictSegmented/", method="POST", json={"samples_list": samples_list, "slice_to_plot": slice_to_plot})
                            if show_segmented_response.status_code == 200:
                                st.success("Predicted segmentations displayed")
                            else:
                                st.error("Failed to show predicted segmentations")
                    except ValueError:
                        st.error("Please provide a valid list of samples")
                else:
                    st.error("Failed to predict segmentation")

    # Model Evaluation Page
    if page == "Model Evaluation":
        st.title("Model Evaluation")
        st.write("Evaluate the segmentation model on the test dataset.")
        
        if st.button("Evaluate Model"):
            response = authenticated_request("/evaluate/", method="POST")
            if response.status_code == 200:
                st.subheader("Model Evaluation Metrics")
                components.html(response.text.replace("<table>", '<table style="color:white;">'), height=400, scrolling=True)
            else:
                st.error("Failed to evaluate model")

    # Drift Detection Page
    if page == "Drift Detection":
        st.title("Drift Detection")
        st.write("Monitor the model for data drift.")

        if st.button("Check for Drift"):
            show_drift()

    # Logout page
    if page == "Logout":
        logout()
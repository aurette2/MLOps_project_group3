import streamlit as st
import requests
from PIL import Image
from io import BytesIO

# Configure the Streamlit app
st.title("Radiology Report Generator")
st.write("Upload a chest X-ray and provide a clinical indication to generate a report.")

# Input field for the authentication token (OAuth2 token)
auth_token = st.text_input("Authentication Token", type="password")

# File uploader for the chest X-ray image
uploaded_file = st.file_uploader("Choose a chest X-ray image...", type=["jpg", "jpeg", "png"])

# Text input for the clinical indication
indication = st.text_input("Clinical Indication", "Patient presenting with persistent cough, fever, and difficulty breathing. Evaluate for pneumonia.")

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Chest X-ray.', use_column_width=True)

    # Call the API to generate the report
    if st.button("Generate Report"):
        if not auth_token:
            st.error("Authentication token is required.")
        else:
            with st.spinner('Generating report...'):
                try:
                    # Prepare the image for sending
                    buffered = BytesIO()
                    image.save(buffered, format="JPEG")
                    img_str = buffered.getvalue()

                    # Call the FastAPI endpoint
                    response = requests.post(
                        "http://127.0.0.1:8000/generate_report/",
                        headers={"Authorization": f"Bearer {auth_token}"},  # Add the token to the header
                        files={"file": ("image.jpg", img_str, "image/jpeg")},
                        data={"indication": indication}
                    )

                    if response.status_code == 200:
                        data = response.json()
                        report = data.get("report", "No report generated.")
                        radiologist = data.get("radiologist_name", "Unknown Radiologist")

                        st.success("Report generated successfully!")
                        st.write(f"**Report:** {report}")
                        st.write(f"**Radiologist:** {radiologist}")
                    else:
                        st.error(f"Error generating the report. Status code: {response.status_code}")
                        st.write(response.text)  # Print the error details

                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

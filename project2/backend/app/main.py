from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.responses import HTMLResponse
from fastapi.security import OAuth2PasswordRequestForm
from datetime import timedelta
from pydantic import BaseModel
import os
from auth import (
    authenticate_user,
    create_access_token,
    ACCESS_TOKEN_EXPIRE_MINUTES,
    decode_token,
    oauth2_scheme
)
# from model import predictByPath, showPredictsById, show_predicted_segmentations, evaluate
from model import Unet
from load_data import Datasource
from eda import DataGenerator
from config import MODELS_DIR, DATASET_BASE_PATH
from elt_report import generate_drift_report
app = FastAPI()

source = Datasource()
train_and_test_ids = source.pathListIntoIds()
test_generator = DataGenerator(source.test_ids)

# Initialize the Unet model (set appropriate parameters)
unet_model = Unet(img_size=128, num_classes=4)
unet_model.compile_and_load_weights( os.path.join(MODELS_DIR,'my_model.keras') )

@app.post("/")
async def hello():
    # Placeholder logic for drift detection
    return {"message": "Welcome"}

# Token endpoint to login and obtain a JWT token
@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"]}, role=user["role"], expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

# # Secured endpoint to retrieve user information
@app.get("/users/me/")
async def read_users_me(token: str = Depends(oauth2_scheme)):
    current_user = decode_token(token)
    return current_user

#Endpoint to get case
@app.get("/case")
async def get_case(num: int = 0):
    return source.test_ids[num][-3:]

# Endpoint to show predictions by ID
@app.get("/showPredictsByID/")
async def show_predicts_by_id(numcase: int, start_slice: int = 60, token: str = Depends(oauth2_scheme)):
    try:
        decode_token(token)
        case = get_case(numcase)
        unet_model.showPredictsById(case, start_slice)
        return {"message": f"Predictions displayed for case: {case}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

#Endpoint to get samples_list
@app.get("/samples_list")
async def get_samples_list():
    return source.test_ids

# Endpoint to show predicted segmented images
@app.post("/showPredictSegmented/")
async def show_predicted_segmentations_api(samples_list: list, slice_to_plot: int, token: str = Depends(oauth2_scheme)):
    try:
        decode_token(token)
        samples_list = get_samples_list()
        unet_model.show_predicted_segmentations(samples_list, slice_to_plot, cmap='gray', norm=None)
        return {"message": "Predicted segmentations displayed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Endpoint to evaluate the model on test data
@app.post("/evaluate/")
async def evaluate_model_api(token: str = Depends(oauth2_scheme)):
    payload = decode_token(token)
    role = payload.get("role")
    
    if role != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not enough permissions")
    
    try:
        # Evaluate the model
        decode_token(token)
        results, descriptions = unet_model.evaluate(test_generator)
        print(results)
        # Create HTML table
        table_html = """
        <html>
            <head>
                <style>
                    table, th, td {
                        border: 1px solid black;
                        border-collapse: collapse;
                        padding: 10px;
                    }
                    th {
                        background-color: #f2f2f2;
                    }
                </style>
            </head>
            <body>
                <table style="color:white;">
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
        """

        # Loop over the results and descriptions to populate the table
        for metric, description in zip(results, descriptions):
            table_html += f"<tr><td>{description}</td><td>{round(metric, 4)}</td></tr>"

        # Close the table and HTML tags
        table_html += """
                </table>
            </body>
        </html>
        """
        print(table_html)
        # Return the HTML content
        return table_html
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Endpoint to predict brain segmentation from image file path
@app.post("/predict/")
async def predict(case_path: str, case: str, token: str = Depends(oauth2_scheme)):
    try:
        username = decode_token(token)
        prediction = unet_model.predictByPath(case_path, case)  # Call method using the instance
        return {"prediction": prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint to show drift (placeholder)
@app.get("/showdrift/")
async def show_drift(token: str = Depends(oauth2_scheme)):
    payload = decode_token(token)
    role = payload.get("role")
    
    if role != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not enough permissions")
    
    try:
        # report_html_path = "custom_report.html"
        report_html_path = DATASET_BASE_PATH + "drift_report.html"

    # Check if the file exists
        if os.path.exists(report_html_path):
            # Read the HTML file and return as a response
            with open(report_html_path, "r") as f:
                html_content = f.read()
            return HTMLResponse(content=html_content)
        else:
            # Generate It from data / This should take time
            generate_drift_report()
            with open(report_html_path, "r") as f:
                html_content = f.read()
            return HTMLResponse(content=html_content)
        
    except Exception as e:
        return {"error": str(e)}

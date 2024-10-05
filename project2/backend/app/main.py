import os
from typing import List


from app.elt_report import generate_drift_report

from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.responses import HTMLResponse
from datetime import timedelta
from pydantic import BaseModel

from app.auth import (
    authenticate_user,
    create_access_token,
    ACCESS_TOKEN_EXPIRE_MINUTES,
    decode_token,
    oauth2_scheme
)
# from model import predictByPath, showPredictsById, show_predicted_segmentations, evaluate
from app.model import Unet
from dotenv import load_dotenv

# Load the environment variables from the .env file
load_dotenv()

app = FastAPI()

# Initialize the Unet model (set appropriate parameters)
unet_model = Unet(img_size=128, num_classes=4)

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
        data={"sub": user["username"]}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

# # Secured endpoint to retrieve user information
@app.get("/users/me/")
async def read_users_me(token: str = Depends(oauth2_scheme)):
    current_user = decode_token(token)
    return current_user


# Endpoint to show predictions by ID
@app.post("/showPredictsByID/")
def show_predicts_by_id(case: str, start_slice: int = 60, token: str = Depends(oauth2_scheme)):
    try:
        username = decode_token(token)
        unet_model.showPredictsById(case, start_slice)  # Use the unet_model instance
        return {"message": f"Predictions displayed for case: {case}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Endpoint to show predicted segmented images
# @app.post("/showPredictSegmented/")
# async def show_predicted_segmentations_api(samples_list: list, slice_to_plot: int, token: str = Depends(oauth2_scheme)):
#     try:
#         username = decode_token(token)
#         unet_model.show_predicted_segmentations(samples_list, slice_to_plot, cmap='gray', norm=None)  # Use the instance
#         return {"message": "Predicted segmentations displayed"}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


# Endpoint to evaluate the model on test data
@app.post("/evaluate/")
def evaluate_model_api(token: str = Depends(oauth2_scheme)):
    try:
        # Evaluate the model (already defined in model.py)
        username = decode_token(token)
        results, descriptions = unet_model.evaluate()
        metrics = {descriptions[i]: results[i] for i in range(len(results))}
        return metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Endpoint to predict brain segmentation from image file path
# @app.post("/predict/")
# async def predict(case_path: str, case: str, token: str = Depends(oauth2_scheme)):
#     try:
#         username = decode_token(token)
#         prediction = unet_model.predictByPath(case_path, case)  # Call method using the instance
#         return {"prediction": prediction.tolist()}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# @app.post("/predict/")
# async def predict(file: UploadFile = File(...)):
#     try:
#         # Decode the token to retrieve the username or relevant user data
#         # username = decode_token(token)

#         # Save the uploaded .nii file to a temporary directory
#         temp_file_path = f"./{file.filename}"
#         print(temp_file_path)
#         with open(temp_file_path, "wb") as f:
#             content = await file.read()
#             f.write(content)

#         # Call the prediction method with the path to the saved file
#         prediction = unet_model.predictFromFile(temp_file_path)  # New method adapted for file input

#         # Return the prediction as a list (to handle numpy arrays)
#         return {"prediction": prediction.tolist()}
    
#     except Exception as e:
#         # If any error occurs, raise an HTTP exception with the error details
#         raise HTTPException(status_code=500, detail=str(e))

# ______________________________________________________________________________________________________________________

@app.post("/predict/")
async def predict(files: List[UploadFile] = File(...)):
    try:
        # Check if exactly two files are uploaded
        if len(files) != 2:
            raise HTTPException(status_code=400, detail="Please upload exactly two files.")

        # Initialize variables to store file paths for flair and t1ce images
        flair_file_path = None
        t1ce_file_path = None

        # Iterate over the uploaded files and verify their filenames
        for file in files:
            if file.filename.endswith("_flair.nii"):
                flair_file_path = f"./{file.filename}"
                with open(flair_file_path, "wb") as f:
                    content = await file.read()
                    f.write(content)
            elif file.filename.endswith("_t1ce.nii"):
                t1ce_file_path = f"./{file.filename}"
                with open(t1ce_file_path, "wb") as f:
                    content = await file.read()
                    f.write(content)
            else:
                raise HTTPException(status_code=400, detail="File names must end with '_flair.nii' and '_t1ce.nii'.")

        # Ensure both flair and t1ce files were uploaded
        if not flair_file_path or not t1ce_file_path:
            raise HTTPException(status_code=400, detail="Both _flair.nii and _t1ce.nii files must be provided.")

        # Call the prediction method with the paths to both files
        prediction = unet_model.predictFromFiles(flair_file_path, t1ce_file_path)

        # Return the prediction as a list (to handle numpy arrays)
        return {"prediction": prediction.tolist()}
    
    except Exception as e:
        # If any error occurs, raise an HTTP exception with the error details
        raise HTTPException(status_code=500, detail=str(e))
    
    
@app.post("/showPredictSegmented/")
async def show_predicted_segmentations_api(files: List[UploadFile] = File(...)):
    try:
        # Check if exactly two files are uploaded
        if len(files) != 2:
            raise HTTPException(status_code=400, detail="Please upload exactly two files.")

        # Initialize variables to store file paths for flair and t1ce images
        flair_file_path = None
        t1ce_file_path = None

        # Iterate over the uploaded files and verify their filenames
        for file in files:
            if file.filename.endswith("_flair.nii"):
                flair_file_path = f"./{file.filename}"
                with open(flair_file_path, "wb") as f:
                    content = await file.read()
                    f.write(content)
            elif file.filename.endswith("_t1ce.nii"):
                t1ce_file_path = f"./{file.filename}"
                with open(t1ce_file_path, "wb") as f:
                    content = await file.read()
                    f.write(content)
            else:
                raise HTTPException(status_code=400, detail="File names must end with '_flair.nii' and '_t1ce.nii'.")

        # Ensure both flair and t1ce files were uploaded
        if not flair_file_path or not t1ce_file_path:
            raise HTTPException(status_code=400, detail="Both _flair.nii and _t1ce.nii files must be provided.")

        # Call the prediction method with the paths to both files
        prediction = unet_model.show_predicted_segmentations(flair_file_path, t1ce_file_path, 60)
        return {"prediction": prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

# ______________________________________________________________________________________________________________________



# Endpoint to show drift (placeholder)
@app.get("/showdrift/")
async def show_drift(token: str = Depends(oauth2_scheme)):
    username = decode_token(token)
    # Placeholder logic for drift detection
    return {"message": "No drift detected (this is a placeholder)"}


@app.get('/monitoring')
async def monitoring():

    try:
        # report_html_path = "custom_report.html"
        
        report_html_path = "../../dataops/brain_data/drift_seg_report.html"

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


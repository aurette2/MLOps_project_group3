from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
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
@app.get("/showPredictsByID/")
def show_predicts_by_id(case: str, start_slice: int = 60, token: str = Depends(oauth2_scheme)):
    try:
        username = decode_token(token)
        unet_model.showPredictsById(case, start_slice)  # Use the unet_model instance
        return {"message": f"Predictions displayed for case: {case}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Endpoint to show predicted segmented images
@app.post("/showPredictSegmented/")
async def show_predicted_segmentations_api(samples_list: list, slice_to_plot: int, token: str = Depends(oauth2_scheme)):
    try:
        username = decode_token(token)
        unet_model.show_predicted_segmentations(samples_list, slice_to_plot, cmap='gray', norm=None)  # Use the instance
        return {"message": "Predicted segmentations displayed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
    username = decode_token(token)
    # Placeholder logic for drift detection
    return {"message": "No drift detected (this is a placeholder)"}

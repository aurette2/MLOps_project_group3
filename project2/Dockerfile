# Use an official Python image as a base
FROM python:3.12

# Set the working directory in the container
WORKDIR /app

# Install necessary system dependencies, including libGL for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0
    
# Copy the requirements file into the container
COPY backend/requirements.txt .

# Install the required Python packages
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the entire project directory into the container
COPY /backend/app /app

# Set up environment variables for Azure authentication
# These variables should be set as secrets in CI/CD pipelines
# or Docker compose files, not hardcoded in the Dockerfile
ENV AZURE_STORAGE_ACCOUNT=<your-azure-storage-account>
ENV AZURE_STORAGE_KEY=<your-azure-storage-key>

# Pull the dataset tracked by DVC
RUN dvc pull

# Expose the ports for FastAPI (8000) and Streamlit (8501)
EXPOSE 8000
EXPOSE 8501

# Set environment variables
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Copy the .env file to the container
COPY /backend/.env .env
COPY /backend/app/starter_code.sh /app
COPY /frontend/app.py /app

# Make the script executable
RUN chmod +x /app/starter_code.sh

# Run the FastAPI app using the built-in `fastapi run` command
CMD ["./starter_code.sh"]

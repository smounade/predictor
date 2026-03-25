# Use official Python image as base
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy your FastAPI app code into the container
COPY . .

# Expose the port FastAPI runs on
EXPOSE 8000

# Command to run the app with uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

# Use the official Python image from the Docker Hub
FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED 1

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt /app/

# Install the required packages
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN python -m pip install scikit_network==0.31.0

# Copy the rest of the application code into the container
COPY . /app/

# Define the command to run the application
CMD ["flask", "run", "--host=0.0.0.0", "--port=5023"]

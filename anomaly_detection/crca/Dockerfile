# Use the official Python image from the Docker Hub
FROM python:3.12

# Set environment variables
ENV PYTHONUNBUFFERED 1

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the required packages
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m pip install --no-cache-dir scikit_network==0.31.0

# Copy the rest of the application code into the container
COPY . .

EXPOSE 5023
# Define the command to run the application
CMD ["flask", "run", "--host=0.0.0.0", "--port=5023"]

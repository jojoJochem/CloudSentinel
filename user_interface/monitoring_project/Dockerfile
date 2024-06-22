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
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . /app/

# Collect static files
RUN python manage.py collectstatic --noinput

# Define the command to run the application
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "monitoring_project.wsgi:application"]

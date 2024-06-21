# Use the official Python image from the Docker Hub
FROM python:3.12-slim AS builder

# Set environment variables
ENV PYTHONUNBUFFERED 1

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt /app/

# Upgrade pip and install the required packages
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . /app/

# Collect static files (if necessary)
RUN python manage.py collectstatic --noinput

# Final stage
FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED 1

# Set the working directory
WORKDIR /app

# Copy installed packages from builder stage
COPY --from=builder /usr/local/lib/python3.8/site-packages /usr/local/lib/python3.8/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy the rest of the application code into the container
COPY . /app/

# Define the command to run the application
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "monitoring_project.wsgi:application"]

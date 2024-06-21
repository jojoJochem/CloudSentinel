# Stage 1: Build stage
FROM python:3.12 as builder

# Set environment variables
ENV PYTHONUNBUFFERED 1

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt /app/

# Install the required packages
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir scikit_network==0.31.0

# Copy the rest of the application code into the container
COPY . /app/

# Stage 2: Final stage
FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED 1

# Set the working directory
WORKDIR /app

# Copy only the necessary files from the build stage
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /app /app

# Define the command to run the application
CMD ["flask", "run", "--host=0.0.0.0", "--port=5023"]

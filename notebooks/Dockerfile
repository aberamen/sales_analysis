# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the application code into the container
COPY . .

# Expose the port your app runs on (if it's a web app)
EXPOSE 5000

# Default command to run unit tests (can be overridden for other purposes)
CMD ["python", "-m", "unittest", "discover", "-s", "tests"]

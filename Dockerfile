# Use the official Python image from the Docker Hub
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project into the container
COPY . .

# Expose port 8000 for the Django app
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DJANGO_SETTINGS_MODULE=Frontend.settings

# Run the Django server
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "Frontend.wsgi:application"]

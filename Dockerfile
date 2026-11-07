# Use official Python image as a base
FROM python:3.11-slim

# Unbuffered output so prints and input() prompts show up immediately in the container
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Set working directory
WORKDIR /app

# Install system dependencies, including pytest.
COPY requirements.txt .
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

# Copy the application modules, UI and automated tests
COPY . .

# Expose the port that the app runs on
EXPOSE 8080

# Command to run the application
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8080", "app:app"]
# Use the official slim Python image as a base
FROM python:3.11-slim

# Unbuffered output so prints and input() prompts show up immediately in the container.
# No .pyc files are written, which keeps the image clean.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Set the working directory
WORKDIR /app

# Install the Python dependencies (Flask, gunicorn, google-genai, pytest, ...).
# requirements.txt is copied first so Docker can cache this layer and only reinstall when the dependencies change.
COPY requirements.txt .
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

# Copy the application modules, templates, static files and automated tests
COPY . .

# Expose the port that the web app runs on
EXPOSE 8080

# Run the web app with gunicorn.
# One worker process, four threads: plan jobs run in the background and their
# status is kept in memory, so every request must reach the same process.
# Threads handle several users at once. Do not raise -w above 1.
CMD ["gunicorn", "-w", "1", "--threads", "4", "-b", "0.0.0.0:8080", "app:app"]
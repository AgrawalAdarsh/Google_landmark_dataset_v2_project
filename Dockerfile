# Use lightweight Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 5000

# Start app with Gunicorn (better for production than Flask dev server)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app.main:app"]

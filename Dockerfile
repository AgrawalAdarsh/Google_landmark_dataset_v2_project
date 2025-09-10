# Use official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements first (for caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download model during build
ENV MODEL_URL="https://drive.google.com/uc?export=download&id=1UGfgPYFZvwq3jmDpfTNJ65nQKFQzpGFa"
RUN apt-get update && apt-get install -y wget unzip && \
    wget -O final_model.keras "$MODEL_URL"

# Copy the rest of the app
COPY . .

# Expose port for Flask
EXPOSE 5000

# Start the app with Gunicorn
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:5000", "app.main:app"]

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install dependencies first (better Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Expose Streamlit's default port
EXPOSE 8501

# Healthcheck so Docker knows if the app is running
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run the Streamlit app
CMD ["python", "-m", "streamlit", "run", "app/dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
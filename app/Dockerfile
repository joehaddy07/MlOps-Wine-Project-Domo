# Use an official Python base image
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Copy only the app's requirements first (for Docker layer caching)
COPY app/requirements.txt .

# Install dependencies using CPU index to force lightweight PyTorch wheels
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

# Copy only the app directory's code into the image
COPY app/ .

# Expose Streamlit's default port
EXPOSE 8501

# Run Streamlit using python -m to guarantee PATH resolution
CMD ["python", "-m", "streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
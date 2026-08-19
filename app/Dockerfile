FROM python:3.12-slim

WORKDIR /app

# Copy requirements from root build context
COPY requirements.txt .

# Install dependencies into system Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy remaining project files
COPY . .

EXPOSE 8501

# Entrypoint to launch Streamlit
CMD ["streamlit", "run", "app/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
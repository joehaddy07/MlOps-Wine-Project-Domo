FROM python:3.12-slim

WORKDIR /app

# Install dependencies into system Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

# Use python -m streamlit run or streamlit run directly
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
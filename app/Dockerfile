FROM python:3.12-slim

WORKDIR /app

COPY app/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY app/apps.py .

COPY app/wine_quality_model.pkl .

EXPOSE 8501

CMD ["streamlit", "run", "apps.py", "--server.port=8501", "--server.address=0.0.0.0"]
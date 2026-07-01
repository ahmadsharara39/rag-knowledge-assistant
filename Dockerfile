FROM python:3.11-slim

WORKDIR /app

# Install core dependencies first for better layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# Runs offline by default; provide provider keys via environment to go live.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

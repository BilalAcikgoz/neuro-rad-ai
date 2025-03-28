FROM python:3.10-slim
WORKDIR /app

# Independents
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Enviroments variables
ENV FLASK_APP=app/main.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=8000

# Flask app run
CMD ["flask", "run"]

FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY agent/     agent/
COPY clients/   clients/
COPY optimizer/ optimizer/
COPY app.py     .

EXPOSE 8000

# CMD ["streamlit", "run", "app.py", \
#      "--server.port=8000", \
#      "--server.address=0.0.0.0", \
#      "--server.headless=true", \
#      "--browser.gatherUsageStats=false"]

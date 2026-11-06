FROM python:3.13-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV GRADER_WORKERS=2
CMD ["sh", "-c", "mkdir -p /data 2>/dev/null; if [ -w /data ]; then export FALLFEST_DB=/data/fallfest.db; else export FALLFEST_DB=/tmp/fallfest.db; fi; exec uvicorn server:app --host 0.0.0.0 --port ${PORT:-10000}"]

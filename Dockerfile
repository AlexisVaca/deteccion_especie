FROM python:3.8-slim
WORKDIR /app
COPY requirements.txt requirements
COPY app.py app.py
COPY model/best.pt model/best.pt

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000
CMD ["gunicorn","-w","4","-b","0.0.0.0:5000","app.app"]

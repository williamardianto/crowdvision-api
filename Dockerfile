FROM python:3.6
ADD . /app
WORKDIR /app
RUN pip install flask gunicorn PIL torch facenet_pytorch pymongo
EXPOSE 8000
CMD ["gunicorn", "-b", "0.0.0.0:8000", "app"]

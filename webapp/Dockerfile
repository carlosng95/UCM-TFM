# FROM python:3.9 as first_stage
FROM python:3.9 as first_stage

WORKDIR /src

COPY requirements.txt /src

RUN pip install -r requirements.txt

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

FROM first_stage as second_stage

COPY /src /src

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
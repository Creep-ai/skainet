FROM python:3.11
RUN set -ex \
    && apt-get update \
    && apt-get --yes autoremove \
    && apt-get --yes clean

WORKDIR /app
COPY . .

RUN pip install -r requirements.txt

CMD streamlit run app.py
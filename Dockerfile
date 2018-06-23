FROM python:3.6.5

ENV GCSFUSE_REPO gcsfuse-jessie
RUN apt-get update && apt-get install --yes --no-install-recommends \
    ca-certificates \
    curl \
  && echo "deb http://packages.cloud.google.com/apt $GCSFUSE_REPO main" \
    | tee /etc/apt/sources.list.d/gcsfuse.list \
  && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - \
  && apt-get update \
  && apt-get install --yes gcsfuse \
  && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

ADD . /app
WORKDIR /app
RUN pip install -r requirements.txt
RUN ["chmod", "+x", "run_will.py"]
RUN ["chmod", "+x", "run_will_docker.sh"]
CMD ./run_will_docker.sh
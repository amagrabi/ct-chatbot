#!/bin/bash

# The steps are kind of boring right now, here's an attempt to automate it.

currentdir=$(pwd)
repodir="$(dirname $(cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd ))"

cd $repodir
rm -rf models
mkdir models
gsutil -m rsync -r gs://ctp-playground-ml-datasets/hipchat/models models
docker build -t gcr.io/commercetools-platform/expert-bot:$(git rev-parse HEAD) -t gcr.io/commercetools-platform/expert-bot:latest .
docker push gcr.io/commercetools-platform/expert-bot

cd $repodir/deployment
helm delete --purge ct-chatbot
helm upgrade --install --wait --set image.tag=latest --namespace expert-bot -f values.yaml -f secrets.yaml ct-chatbot ct-chatbot

cd $currentdir
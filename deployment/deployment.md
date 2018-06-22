# Packaging

## create the docker container

Right now the models are baked into the Docker image...

```
gsutil -m rsync -r gs://ctp-playground-ml-datasets/hipchat/models models

docker build -t gcr.io/commercetools-platform/expert-bot:$(git rev-parse HEAD) -t gcr.io/commercetools-platform/expert-bot:latest .
```

## push the docker container

```
docker push gcr.io/commercetools-platform/expert-bot
```

# Deployment

## Redis on K8S

```
helm install --namespace expert-bot --set usePassword=false --name ct-chatbot-redis stable/redis
```

helm upgrade --install --namespace expert-bot --set usePassword=false ct-chatbot-redis .

## bot
cd deployment

```
helm upgrade --install --wait --set image.tag=$(git rev-parse HEAD) --namespace expert-bot -f values.yaml -f secrets.yaml ct-chatbot ct-chatbot
```

helm upgrade --install --wait --namespace expert-bot -f values.yaml -f secrets.yaml ct-chatbot ct-chatbot


## remove deployment
```
helm delete --purge ct-chatbot
```

#!/bin/sh

docker build -t deniskocs/ai:llm-service .
docker push deniskocs/ai:llm-service
echo "Image built and pushed successfully!"

#!/bin/sh
docker run --env-file settings.env -p 10.0.0.8:8090:8090 -u 1000:1000 --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v /home/denis/data/cache:/.cache deniskocs/ai:llm-service

#--entrypoint= --rm  -p 10.0.0.8:8090:8090
echo "Container started successfully!"

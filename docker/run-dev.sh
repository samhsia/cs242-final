#!/bin/bash

docker run \
  --security-opt seccomp=/group/vlsiarch/schsia/seccomp/default.json \
  --gpus all \
  -v /home/schsia:/home/schsia:rw \
  -v /project/recommendation:/project/recommendation:rw \
  -v /home/schsia/.cache:/home/schsia/.cache:ro \
  -v /group/brooks:/group/brooks:ro \
  -v /group/vlsiarch:/group/vlsiarch:rw \
  -v /etc/passwd:/etc/passwd:ro \
  -v /data/:/data/:rw \
  -p 8888:8888 \
  -it --rm --user $(id -u) 242:gpu  /bin/bash -l

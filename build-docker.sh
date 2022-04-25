#!/bin/sh
docker build -t gcr.io/viewo-g/piper/agent/cpu/roberto_utils:3 -f Dockerfile .
docker push gcr.io/viewo-g/piper/agent/cpu/roberto_utils:3


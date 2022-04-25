FROM gcr.io/viewo-g/piper/agent/runner/cpu/main:1.43.6.latest

COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip
RUN pip install -r /tmp/requirements.txt
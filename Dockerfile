FROM python:latest

RUN apt-get update && apt-get install -y portaudio19-dev python3-pyaudio
ADD requirements.txt .
RUN pip install -r requirements.txt

ADD audio_levels.py .

CMD ./audio_levels.py

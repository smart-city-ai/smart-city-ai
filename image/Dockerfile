From jjanzic/docker-python3-opencv:contrib-opencv-3.4.2

RUN pip3 install flask jsonpickle numpy imutils requests

COPY match.py extract.py server.py server.* /opt/images/
ENV FLASK_APP server.py
WORKDIR /opt/images
ENTRYPOINT ["flask" "run"]

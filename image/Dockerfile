From jjanzic/docker-python3-opencv:contrib-opencv-3.4.2

RUN pip3 install flask jsonpickle numpy imutils requests

COPY match.py extract.py server.py server.* /opt/images/
WORKDIR /opt/images
ENTRYPOINT ["python3"]
CMD ["server.py"]

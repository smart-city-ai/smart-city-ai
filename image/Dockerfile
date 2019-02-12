From fedora:28
RUN dnf install python3-opencv opencv-contrib python3-pip -y
RUN pip3 install flask jsonpickle numpy imutils requests

COPY match.py extract.py server.py server.* /opt/images/
WORKDIR /opt/images
ENTRYPOINT ["python3"]
CMD ["server.py"]

FROM ubuntu:latest
LABEL authors="hara"

ENTRYPOINT ["top", "-b"]
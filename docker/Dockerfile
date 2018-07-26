FROM pytorch:latest

RUN apt-get update && apt-get install -y --no-install-recommends \
	build-essential \
	libboost-python-dev \
	libexpat1-dev \
	zlib1g-dev \
	libbz2-dev \
	libgtk2.0-dev \
	libspatialindex-dev

ADD . /robo-src
WORKDIR /robo-src
RUN pip install -r deps/requirements-lock.txt

RUN python3 -m pip uninstall -y osmium
RUN rm -rf ~/.cache/pip/wheels/
RUN rm /usr/lib/x86_64-linux-gnu/libboost_python.so
RUN ln -s /usr/lib/x86_64-linux-gnu/libboost_python-py35.so /usr/lib/x86_64-linux-gnu/libboost_python.so
RUN yes | python3 -m pip install osmium

ENTRYPOINT ["./rs"]
CMD ["--help"]

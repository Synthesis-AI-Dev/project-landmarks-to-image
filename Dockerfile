FROM python:3.8

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y \
	libglib2.0-0 \
 	libopenexr-dev \
	libsm6 \
	libxext-dev \
	libxrender1 \
	zlib1g-dev \
 && rm -rf /var/lib/apt/lists/*

# Copy requirements first to use layer caching
COPY requirements.txt /
RUN pip install --no-cache-dir -r /requirements.txt

COPY . /opt/project-landmark-to-image
WORKDIR /opt/project-landmark-to-image
RUN chmod a+x *.py
CMD bash

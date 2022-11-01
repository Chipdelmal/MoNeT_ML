FROM python:3.8-slim-buster
LABEL maintainer="Hector M. Sanchez C. <sanchez.hmsc@berkeley.edu>"

###############################################################################
# Update and install linux packages
###############################################################################
RUN apt-get update \
    && apt-get install nano

###############################################################################
# Install python requirements
###############################################################################
RUN mkdir requirements
COPY app/requirements.txt ./requirements/
RUN pip install --upgrade pip\
    && pip install -r ./requirements/requirements.txt --no-cache-dir

###############################################################################
# Copy files
###############################################################################
COPY app/ ./

###############################################################################
# Run server
###############################################################################
CMD [ "gunicorn", "--workers=5", "--threads=1", "-b 0.0.0.0:80", "app:server"]
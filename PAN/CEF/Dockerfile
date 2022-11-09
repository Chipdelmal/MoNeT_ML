FROM python:3.10-slim-buster
LABEL maintainer="Hector M. Sanchez C. <sanchez.hmsc@berkeley.edu>"

ENV DASH_DEBUG_MODE False
###############################################################################
# Update and install linux packages
###############################################################################
RUN apt-get update \
    && pip install --upgrade pip
###############################################################################
# Install python requirements
###############################################################################
RUN mkdir requirements
COPY app/requirements.txt ./requirements/
RUN set -ex \
    && pip install -r ./requirements/requirements.txt --no-cache-dir

###############################################################################
# Copy files
###############################################################################
COPY app/ ./

###############################################################################
# Run server
###############################################################################
EXPOSE 8050
CMD ["gunicorn", "-b", "0.0.0.0:8050", "--reload", "app:server"]
FROM python:3.10-slim-buster
LABEL maintainer="Hector M. Sanchez C. <sanchez.hmsc@berkeley.edu>"

ENV DASH_DEBUG_MODE False
###############################################################################
# Update and install linux packages
###############################################################################
RUN apt-get update \
    && apt-get -y install gcc mono-mcs \
    && rm -rf /var/lib/apt/lists/* \
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
EXPOSE 8000
CMD ["gunicorn", "-b", "0.0.0.0:8000", "--reload", "app:server"]
# CMD ["python", "app.py"]
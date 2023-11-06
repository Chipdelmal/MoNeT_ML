#!/bin/sh

###############################################################################
# Freeze Requirements
###############################################################################
condaFreeze:
	- pip freeze > ./app/requirements.txt
	- conda env export | grep -v "prefix" > ./app/requirements.yml

###############################################################################
# Docker Local
###############################################################################
dockerBuild:
	- docker rmi monet-pgs:dev -f
	- docker build -t monet-pgs:dev .

dockerRun:
	- docker run -p 8000:8000 -v ${PWD}/app:/app --rm monet-pgs:dev

###############################################################################
# Docker Release
###############################################################################
VERSION=$(file < ./app/version.txt)

dockerVersion:
	- printf "$(VERSION)"

dockerRelease:
	- docker buildx build . \
		--platform=linux/amd64,linux/arm64 \
		-t chipdelmal/monet-pgs:$(VERSION) \
		-t chipdelmal/monet-pgs:latest \
		--push

dockerPull:
	- docker pull chipdelmal/monet-pgs:$(VERSION)

###############################################################################
# Heroku Release
# 	https://samedwardes.com/2019/11/15/dash-heroku-cookie-cutter/
#	https://github.com/SamEdwardes/dash-heroku-cookie-cutter
###############################################################################
herokuRelease:
	- docker image build -t dash-heroku:latest .
	- heroku container:push web --app monet-pgs
	- heroku container:release web --app monet-pgs
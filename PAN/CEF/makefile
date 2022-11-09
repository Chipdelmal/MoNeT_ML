
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
	- docker rmi monet-cef:dev -f
	- docker build -t monet-cef:dev .

dockerRun:
	- docker run -p 8050:8050 -v "$(pwd)"/app:/app --rm monet-cef:dev

###############################################################################
# Docker Release
###############################################################################
VERSION="0.2.0"
dockerRelease:
	- docker rmi chipdelmal/monet-cef:$(VERSION) -f
	- docker build -t chipdelmal/monet-cef:$(VERSION) .
	- docker push chipdelmal/monet-cef:$(VERSION)

dockerPull:
	- docker pull chipdelmal/monet-cef:$(VERSION)

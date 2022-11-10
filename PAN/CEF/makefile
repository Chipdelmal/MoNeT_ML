
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
	- docker run -p 5000:5000 -v "$(pwd)"/app:/app --rm monet-cef:dev

###############################################################################
# Docker Release
###############################################################################
VERSION="0.2.3"
dockerRelease:
	- docker rmi chipdelmal/monet-cef:$(VERSION) -f
	- docker build -t chipdelmal/monet-cef:$(VERSION) .
	- docker push chipdelmal/monet-cef:$(VERSION)

dockerPull:
	- docker pull chipdelmal/monet-cef:$(VERSION)

###############################################################################
# Heroku Release
# 	https://samedwardes.com/2019/11/15/dash-heroku-cookie-cutter/
#	https://github.com/SamEdwardes/dash-heroku-cookie-cutter
###############################################################################
herokuRelease:
	- docker image build -t dash-heroku:latest .
	- heroku container:push web --app monet-cef
	- heroku container:release web --app monet-cef
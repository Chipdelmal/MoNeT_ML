
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
	- docker run -p 8080:8080 -v "$(pwd)"/app:/app --rm monet-pgs:dev

###############################################################################
# Docker Release
###############################################################################
VERSION=$(file < ./app/version.txt)
dockerRelease:
	- docker build -t chipdelmal/monet-pgs:$(VERSION) .
	- docker push chipdelmal/monet-pgs:$(VERSION)
	- docker build -t chipdelmal/monet-pgs:latest .
	- docker push chipdelmal/monet-pgs:latest

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
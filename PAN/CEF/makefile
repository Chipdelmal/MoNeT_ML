condaFreeze:
	- pip freeze > ./app/requirements.txt
	- conda env export | grep -v "prefix" > ./app/requirements.yml

dockerBuild:
	- docker rmi monet_cef:dev -f
	- docker build -t monet_cef:dev .
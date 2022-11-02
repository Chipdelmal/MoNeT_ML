# pgSIT Cost-Effectiveness

## Instructions

Download and install [docker](https://docs.docker.com/get-docker/), then pull our MGDrivE Webinar image from [our dockerhub](https://hub.docker.com/repository/docker/chipdelmal/monet-cef) with:

```bash
docker pull chipdelmal/monet-cef:TAG_NUMBER
```

Replacing the `TAG_NUMBER` with the latest version of the image (eg. `docker pull chipdelmal/monet-cef:0.1.1`).

To run the image, run the following command in the terminal:

```bash
docker run -p 8050:8050 -v "$(pwd)"/app:/app --rm chipdelmal/monet-cef:TAG_NUMBER
```

Again, replacing the `TAG_NUMBER` with the version downloaded in the previous step.

Finally, open the following address on your favorite browser:

```bash
localhost:8050
```

## Sources

* https://github.com/yaojiach/docker-dash
* https://github.com/conducive333/mgdrive-fdml-app
* https://www.devcoons.com/how-to-deploy-your-plotly-dash-dashboard-using-docker/
* https://github.com/atheo89/dashboard-deployment
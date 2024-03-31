## FOR DOCKER DEMO

### GIT CLONE

`git clone --branch task3 https://github.com/heliosgrounder/MLCV.git`

---

### DOCKER BUILD

run `docker build -t <name_of_docker_image>`

wait some time...

---

### RUN DOCKER IMAGE

run `docker run <name_of_docker_image> /bin/bash -c "cd data && streamlit run helios_package/webserver.py"`

---

### OPEN WEBSERVER WITH BROWSER AND USE DEMO

docker image should start and give you URL. Paste this URL in browser and use the demo.
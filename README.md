## FOR DOCKER DEMO

### GIT CLONE

`git clone --branch task3 https://github.com/heliosgrounder/MLCV.git`

---

### DOCKER BUILD

run `docker build -t <name_of_docker_image>`

wait some time...

---

### RUN DOCKER IMAGE

Linux: `docker run <name_of_docker_image> /bin/bash -c "cd data && streamlit run helios_package/webserver.py"`

Windows: `docker run -p 8501:8501 <name_of_docker_image> /bin/bash -c "cd data && streamlit run helios_package/webserver.py --server.port=8501 --server.address=0.0.0.0"`

---

### OPEN WEBSERVER WITH BROWSER AND USE DEMO

`https://localhost:8501/`
docker image should start and give you URL. Paste this URL in browser and use the demo.
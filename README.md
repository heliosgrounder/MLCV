## FOR LOCAL TESTING

### GIT CLONE

`git clone --branch task2 https://github.com/heliosgrounder/MLCV.git`

---

### CREATE VIRTUAL ENVIRONMENT

make sure that you are in the root folder of project

*using venv* `python3 -m venv virtual_env_name`

*using virtualenv* `virtualenv -p /path/to/python3 virtual_env_name`

---

### ACTIVATE VIRTUAL ENVIRONMENT

On Unix `source virtualenvname/bin/activate`

On Windows CMD `virtualenvname/Scripts\activate.bat`

---

### UNPACK PACKAGE

Any platform `tar -xzf helios_package-0.1.38.tar.gz`

---

### GET TO THE ROOT FOLDER

Any platform `cd helios_package-0.1.38`

---

### INSTALL PACKAGE DEPENDENCIES

Any platform `pip install .`

---

### ADD EMPTY GIT REPOSITORY AND ADD FILES

Any platform `git init`

Any platform `git add .`

### RUN PRE-COMMIT

Run `pre-commit run -a`

Check if all steps are passed.

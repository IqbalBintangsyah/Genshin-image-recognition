# Activate python virtual environment
source ./bin/activate

# Build and run docker image
docker build -t test .
docker run -it test /bin/bash
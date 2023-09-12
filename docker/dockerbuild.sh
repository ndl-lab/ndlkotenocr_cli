TAG=kotenocr-cli-py37-worker
DOCKERIGNORE=docker/dockerignore
DOCKERFILE=docker/Dockerfile
cp ${DOCKERIGNORE} .dockerignore
docker build -t ${TAG} -f ${DOCKERFILE} .
rm .dockerignore

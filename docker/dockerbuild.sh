TAG=kotenocr-cli-py310
DOCKERIGNORE=docker/dockerignore
DOCKERFILE=docker/Dockerfile
cp ${DOCKERIGNORE} .dockerignore
docker build -t ${TAG} -f ${DOCKERFILE} .
rm .dockerignore

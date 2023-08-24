SET TAG=kotenocr-cli-py37
SET DOCKERIGNORE=docker\dockerignore
SET DOCKERFILE=docker\Dockerfile

copy %DOCKERIGNORE% .dockerignore
docker build -t %TAG% -f %DOCKERFILE% .
del .dockerignore

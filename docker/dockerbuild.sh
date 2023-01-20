TAG=kotenocr-cli-py37
DOCKERIGNORE=docker/dockerignore
DOCKERFILE=docker/Dockerfile
wget -nc https://lab.ndl.go.jp/dataset/ndlkotensekiocr/trocr/models.zip -P ./src/text_kotenseki_recognition/
wget -nc https://lab.ndl.go.jp/dataset/ndlkotensekiocr/layoutmodel/models.zip -P ./src/ndl_kotenseki_layout/
cp ${DOCKERIGNORE} .dockerignore
docker build -t ${TAG} -f ${DOCKERFILE} .
rm .dockerignore

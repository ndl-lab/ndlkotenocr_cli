TAG=kotenocr-cli-py37
DOCKERIGNORE=docker/dockerignore
DOCKERFILE=docker/Dockerfile
wget -nc https://lab.ndl.go.jp/dataset/ndlkotensekiocr/trocr/model-ver2.zip -P ./src/text_kotenseki_recognition/
wget -nc https://lab.ndl.go.jp/dataset/ndlkotensekiocr/layoutmodel/ndl_kotenseki_layout_ver2.pth -P ./src/ndl_kotenseki_layout/models/
cp ${DOCKERIGNORE} .dockerignore
docker build -t ${TAG} -f ${DOCKERFILE} .
rm .dockerignore

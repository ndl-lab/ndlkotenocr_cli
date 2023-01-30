SET TAG=kotenocr-cli-py37
SET DOCKERIGNORE=docker\dockerignore
SET DOCKERFILE=docker\Dockerfile


set DIRNAME=src\text_kotenseki_recognition\
set FILENAME=models.zip
set URL="https://lab.ndl.go.jp/dataset/ndlkotensekiocr/trocr/models.zip"
set FULLPATH=%DIRNAME%%FILENAME%
if not exist %DIRNAME%%FILENAME% (
mkdir %DIRNAME%
curl -o %FULLPATH% %URL%
)


set DIRNAME=src\ndl_kotenseki_layout\
set FILENAME=models.zip
set URL="https://lab.ndl.go.jp/dataset/ndlkotensekiocr/layoutmodel/models.zip"
set FULLPATH=%DIRNAME%%FILENAME%
if not exist %DIRNAME%%FILENAME% (
mkdir %DIRNAME%
curl -o %FULLPATH% %URL%
)

copy %DOCKERIGNORE% .dockerignore
docker build -t %TAG% -f %DOCKERFILE% .
del .dockerignore

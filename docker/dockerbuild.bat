SET TAG=ocr-cli-py37
SET DOCKERIGNORE=docker\dockerignore
SET DOCKERFILE=docker\Dockerfile


set DIRNAME=src\text_recognition\models\
set FILENAME=ndlenfixed64-mj0-synth1.pth
set URL="https://lab.ndl.go.jp/dataset/ndlocr/text_recognition/ndlenfixed64-mj0-synth1.pth"
set FULLPATH=%DIRNAME%%FILENAME%
if not exist %DIRNAME%%FILENAME% (
mkdir %DIRNAME%
curl -o %FULLPATH% %URL%
)


set DIRNAME=src\ndl_layout\models\
set FILENAME=epoch_140_all_eql_bt.pth
set URL="https://lab.ndl.go.jp/dataset/ndlocr/ndl_layout/epoch_140_all_eql_bt.pth"
set FULLPATH=%DIRNAME%%FILENAME%
if not exist %DIRNAME%%FILENAME% (
mkdir %DIRNAME%
curl -o %FULLPATH% %URL%
)

copy %DOCKERIGNORE% .dockerignore
docker build -t %TAG% -f %DOCKERFILE% .
del .dockerignore
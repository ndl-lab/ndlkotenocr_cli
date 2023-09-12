docker run --gpus 0 -d --add-host=host.docker.internal:host-gateway --rm --name kotenocr_cli_runner_0 -i kotenocr-cli-py37-worker:latest&
docker run --gpus 1 -d --add-host=host.docker.internal:host-gateway --rm --name kotenocr_cli_runner_1 -i kotenocr-cli-py37-worker:latest&
docker run --gpus 2 -d --add-host=host.docker.internal:host-gateway --rm --name kotenocr_cli_runner_2 -i kotenocr-cli-py37-worker:latest&
docker run --gpus 3 -d --add-host=host.docker.internal:host-gateway --rm --name kotenocr_cli_runner_3 -i kotenocr-cli-py37-worker:latest

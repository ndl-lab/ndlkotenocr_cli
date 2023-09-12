# Copyright (c) 2022, National Diet Library, Japan
#
# This software is released under the CC BY 4.0.
# https://creativecommons.org/licenses/by/4.0/


import click
import json
import os
import sys
from celery import Celery

from cli.core import OcrInferencer
from cli.core import utils

brokerurl='amqp://root:password@host.docker.internal'
app = Celery('tasks', broker=brokerurl,backend=brokerurl)
@app.task(bind=True)
def task(message):
    input_root=message
    output_root="/data/celerykotenseki/outputtxt/"
    input_structure="s"
    config_file="config.yml"
    add_info=False

    click.echo('start inference !')
    click.echo('input_root : {0}'.format(input_root))
    click.echo('output_root : {0}'.format(output_root))
    click.echo('config_file : {0}'.format(config_file))
    click.echo('add_info : {0}'.format(add_info))

    cfg = {
        'input_root': input_root,
        'output_root': output_root,
        'config_file': config_file,
        'input_structure': input_structure,
        'add_info': add_info
    }

    # check if input_root exists
    if not os.path.exists(input_root):
        print('INPUT_ROOT not found :{0}'.format(input_root), file=sys.stderr)
        sys.exit(0)

    # parse command line option
    infer_cfg = utils.parse_cfg(cfg)
    if infer_cfg is None:
        print('[ERROR] Config parse error :{0}'.format(input_root), file=sys.stderr)
        sys.exit(1)

    # prepare output root derectory
    #infer_cfg['output_root'] = utils.mkdir_with_duplication_check(infer_cfg['output_root'])

    # save inference option
    with open(os.path.join(infer_cfg['output_root'], 'opt.json'), 'w') as fp:
        json.dump(infer_cfg, fp, ensure_ascii=False, indent=4,
                  sort_keys=True, separators=(',', ': '))
    # do inference
    try:
        inferencer = OcrInferencer(infer_cfg)
        inferencer.run()
    except Exception as e:
        return e


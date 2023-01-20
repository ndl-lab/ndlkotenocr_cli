# Copyright (c) 2022, National Diet Library, Japan
#
# This software is released under the CC BY 4.0.
# https://creativecommons.org/licenses/by/4.0/


import copy
import datetime
import glob
import os
import sys
import yaml


def parse_cfg(cfg_dict):
    """
    コマンドで入力された引数やオプションを内部関数が利用しやすい形にparseします。

    Parameters
    ----------
    cfg_dict : dict
        コマンドで入力された引数やオプションが保存された辞書型データ。

    Returns
    -------
    infer_cfg : dict
        推論処理を実行するための設定情報が保存された辞書型データ。
    """
    infer_cfg = copy.deepcopy(cfg_dict)

    # add inference config parameters from yml config file
    yml_config = None
    if not os.path.isfile(cfg_dict['config_file']):
        print('[ERROR] Config yml file not found.', file=sys.stderr)
        return None

    with open(cfg_dict['config_file'], 'r') as yml:
        yml_config = yaml.safe_load(yml)

    if type(yml_config) is not dict:
        print('[ERROR] Config yml file read error.', file=sys.stderr)
        return None

    infer_cfg.update(yml_config)
    infer_cfg['proc_range'] = {
        'start': 0,
        'end': 2
    }
    infer_cfg['partial_infer'] = False
    # create input_dirs from input_root
    # input_dirs is list of dirs that contain img (and xml) dir
    infer_cfg['input_root'] = os.path.abspath(infer_cfg['input_root'])
    infer_cfg['output_root'] = os.path.abspath(infer_cfg['output_root'])
    if infer_cfg['input_structure'] in ['s']:
        # - Sigle input dir mode
        # input_root
        #  ├── xml
        #  │   └── R[7桁連番].xml※XMLデータ
        #  └── img
        #      └── R[7桁連番]_pp.jp2※画像データ

        # validation check for input dir structure
        if not os.path.isdir(os.path.join(infer_cfg['input_root'], 'img')):
            print('[ERROR] Input img diretctory not found in {}'.format(infer_cfg['input_root']), file=sys.stderr)
            return None
        infer_cfg['input_dirs'] = [infer_cfg['input_root']]
    elif infer_cfg['input_structure'] in ['f']:
        # - Image file input mode
        # input_root is equal to input image file path
        infer_cfg['input_dirs'] = [infer_cfg['input_root']]
    else:
        print('[ERROR] Unexpected input directory structure type: {0}.'.format(infer_cfg['input_structure']), file=sys.stderr)
        return None

    return infer_cfg


def save_xml(xml_to_save, path):
    """
    指定されたファイルパスにXMLファイル保存します。

    Parameters
    ----------
    path : str
        XMLファイルを保存するファイルパス。

    """
    print('### save xml : {}###'.format(path))
    try:
        xml_to_save.write(path, encoding='utf-8', xml_declaration=True)
    except OSError as err:
        print("[ERROR] XML save error : {0}".format(err), file=sys.stderr)
        raise OSError
    return


def mkdir_with_duplication_check(dir_path):
    dir_path_to_create = dir_path

    # prepare output root derectory
    while os.path.isdir(dir_path_to_create):
        print('[WARNING] Directory {0} already exist.'.format(dir_path))
        now = datetime.datetime.now()
        time_stamp = now.strftime('_%Y%m%d%H%M%S')
        dir_path_to_create += time_stamp

    if dir_path_to_create != dir_path:
        print('[WARNING] Directory is changed to {0}.'.format(dir_path_to_create))
    os.mkdir(dir_path_to_create)

    return dir_path_to_create

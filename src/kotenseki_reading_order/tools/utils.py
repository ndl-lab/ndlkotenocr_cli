#!/usr/bin/env python

# Copyright (c) 2022, National Diet Library, Japan
#
# This software is released under the CC BY 4.0.
# https://creativecommons.org/licenses/by/4.0/

import argparse
import inspect
from collections import OrderedDict


def get_rec(d, key):
    def _get_rec(d, ks):
        if d is None or len(ks) == 0:
            return d
        return _get_rec(d.get(ks[0]), ks[1:])
    return _get_rec(d, key.split("."))


def get_list_type(type):
    import re
    match = re.findall("typing.List\[(.*)\]", str(type))  # noqa: W605
    if match:
        return eval(match[0])
    return None


def add_argument(parser, fn, **kwargs):
    assert inspect.isfunction(fn)
    sig = inspect.signature(fn)
    params = sig.parameters
    fn_vars = vars(fn)
    for k, p in params.items():
        kwargs = {}
        kwargs['help'] = get_rec(fn_vars, 'help.{}'.format(k))
        kwargs['choices'] = get_rec(fn_vars, 'choices.{}'.format(k))
        kwargs['default'] = p.default
        type = p.annotation if p.annotation != inspect._empty else None
        kwargs['type'] = type

        if kwargs['default'] != inspect._empty:
            default_str = " (default: {})".format(kwargs['default'])
            if kwargs['help']:
                kwargs['help'] += default_str
            else:
                kwargs['help'] = default_str

        list_type = get_list_type(type)
        if list_type:
            kwargs['type'] = list_type
            kwargs['nargs'] = '+'

        if type is bool:
            if k.startswith('use') or k.startswith('enable') or k.startswith('disable') or k.startswith('ignore') or k.startswith('naive'):
                kwargs = {'action': 'store_true', 'help': kwargs['help']}

        if p.kind == inspect._ParameterKind.POSITIONAL_OR_KEYWORD:
            if kwargs.get('default', inspect._empty) != inspect._empty or kwargs.get('action', "").startswith('store'):
                parser.add_argument('--' + k, **kwargs)
            else:
                parser.add_argument(k, **kwargs)
        elif p.kind == inspect._ParameterKind.KEYWORD_ONLY:
            parser.add_argument('--' + k, **kwargs)


def make_argparser(fn, parser=None, **kwargs):
    if parser is None:
        parser = argparse.ArgumentParser(**kwargs)
    if inspect.isfunction(fn):
        add_argument(parser, fn)
        parser.set_defaults(handler=fn)
    elif isinstance(fn, list):
        subp = parser.add_subparsers()
        for fn_dict in fn:
            if inspect.isfunction(fn_dict):
                fn_dict = {'name': fn_dict.__name__,
                           'description': fn_dict.__doc__, 'func': fn_dict}
            p = subp.add_parser(
                fn_dict['name'], description=fn_dict.get('description', None))
            make_argparser(fn_dict['func'], parser=p)
    elif isinstance(fn, dict):
        fn = [{'name': name, 'description': val.__doc__, 'func': val} if inspect.isfunction(
            val) else {'name': name, 'func': val} for name, val in fn.items()]
        make_argparser(fn, parser=parser)
    else:
        assert False, fn
    return parser


def add_params(val_name, help=None, choices=None):
    def _add_params(fn):
        if not hasattr(fn, 'help'):
            fn.help = {}
        if not hasattr(fn, 'choices'):
            fn.choices = {}
        fn.help[val_name] = help
        fn.choices[val_name] = choices
        return fn
    return _add_params


def auto_run(*args, **kwargs):
    parser = make_argparser(*args, **kwargs)
    args = parser.parse_args()
    if not hasattr(args, 'handler'):
        parser.print_help()
        return
    handler = args.handler
    _args = args._get_args()
    _kwargs = dict(args._get_kwargs())
    del(_kwargs['handler'])
    return handler(*_args, **_kwargs)


def argslist_to_dict(argslist):
    """Convert args list to dictionary.
    It converts ["KEY1=VAL1,KEY2=VAL2", "KEY3=VAL3"]
    to {"KEY1": "VAL1", "KEY2": "VAL2", "KEY3": "VAL3"}
    """
    argsdict = OrderedDict()
    for x in argslist:
        kvs = x.split(',')
        for kv in kvs:
            eq = kv.find('=')
            k, v = (kv[:eq].strip() if 0 <= eq else '', kv[eq+1:].strip())
            argsdict[k] = v
    return argsdict
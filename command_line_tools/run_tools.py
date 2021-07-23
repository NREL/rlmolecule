import errno
import json
import os
import sys
import time
from pprint import pprint
from typing import Optional
from . import command_line_config


def makedir_if_not_exists(filename: str) -> None:
    try:
        os.makedirs(filename)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


log_path = os.path.join(os.path.curdir, 'log')
makedir_if_not_exists(log_path)


def setup_run(
        default_config: {},
        subrun_name: Optional[str] = None,
) -> ({}, str):
    '''
    Sets up a config and logging prefix for this run. Writes the config to a log file.
    :param default_config:
    :param subrun_name:
    :return:
    '''
    config = command_line_config.parse_config_from_args(sys.argv[1:], default_config)
    pprint(config)
    run_name = get_run_name(config, subrun_name)
    config['run_name'] = run_name
    write_config_log(config)
    return config, run_name


def write_config_log(
        config: {},
        run_name: Optional[str] = None,
        suffix: str = '_config',
        path: Optional[str] = None,
) -> None:
    '''
    Writes a json log file containing the configuration of this run
    :param config: run config
    :param run_name: run name
    :param suffix: suffix to append to run name to get the config log file name
    :param path: where to store the config log file. If None, the current directory + '/log' will be used.
    '''
    if run_name is None:
        if 'run_prefix' in config:
            run_name = config['run_prefix']
        elif 'name' in config:
            run_name = config['name']
        else:
            run_name = 'run'

    if path is None:
        path = os.path.join(os.path.curdir, 'log')

    makedir_if_not_exists(path)

    config_filename = os.path.join(path, run_name + suffix + '.json')
    with open(config_filename, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2, sort_keys=True)


def get_run_name(
        config: {},
        subrun_name: Optional[str] = None,
) -> str:
    '''
    Makes the run log file prefix string by concatenating the run name, subrun name (if it exists), and the current
    time
    :param config: run config
    :param subrun_name: a subrun name to optionally append to the run name
    :return: run log file prefix
    '''
    start_time = int(time.time() * 10000)
    subrun_name = '' if subrun_name is None else '_' + subrun_name
    return config['name'] + subrun_name + '_' + str(start_time) + '_'

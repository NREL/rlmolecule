""" Utilities for loading training and game parameters, and for setting up the sql database
"""

import os

import yaml
from sqlalchemy import create_engine

# TODO add the command line args that correspond to the config file options here


class RunConfig:
    def __init__(self, config_file, **kwargs):

        self.config_map = {}
        if config_file is not None:
            with open(config_file, 'r') as f:
                # self.config_map = yaml.safe_load(f)
                # expandvars is a neat trick to expand bash variables within the yaml file
                # from here: https://stackoverflow.com/a/60283894/7483950
                self.config_map = yaml.safe_load(os.path.expandvars(f.read()))

        # TODO overwrite settings in the config file if they were passed in via kwargs
        # Settings for setting up scripts to run everything
        # self.run_config = self.config_map.get('run_config',{})
        self.run_id = self.config_map.get('run_id', 'test')

        # Settings specific to the problem at hand
        self.problem_config = self.config_map.get('problem_config', {})
        # Settings for training the policy model
        self.train_config = self.config_map.get('train_config', {})
        self.mcts_config = self.config_map.get('mcts_config', {})

    # def load_config_file(config_file):
    #     with open(config_file, 'r') as conf:
    #         config_map = yaml.load(conf)

    #     return config_map

    def start_engine(self):
        self.engine = RunConfig.start_db_engine(**self.config_map.get('sql_database', {}))
        return self.engine

    @staticmethod
    def start_db_engine(**kwargs):
        """ Connect to the sql database that will store the game and reward data
        used by the policy model and game runner
        """
        drivername = kwargs.get('drivername', 'sqlite')
        db_file = kwargs.get('db_file', 'game_data.db')
        if drivername == 'sqlite':
            engine = create_engine(
                f'sqlite:///{db_file}',
                # The 'check_same_thread' option only works for sqlite
                connect_args={'check_same_thread': False},
                execution_options={"isolation_level": "AUTOCOMMIT"})
        else:
            engine = RunConfig.start_server_db_engine(**kwargs)

        return engine

    @staticmethod
    def start_server_db_engine(drivername="postgresql+psycopg2",
                               dbname='bde',
                               port=None,
                               host=None,
                               user=None,
                               passwd_file=None,
                               passwd=None,
                               **kwargs):
        # By default, use the host defined in the environment
        host = os.getenv('DB_HOST', host)

        # add the ':' to separate host from port
        port = ":" + str(port) if port is not None else ""

        if passwd_file is not None:
            # read the password from a file
            with open(passwd_file, 'r') as f:
                passwd = f.read().strip()
        # add the ':' to separate user from passwd
        passwd = ":" + str(passwd) if passwd is not None else ""

        engine_str = f'{drivername}://{user}{passwd}@{host}{port}/{dbname}'
        # don't print since it has the user's password
        # print(f'connecting to database using: {engine_str}')
        engine = create_engine(engine_str, execution_options={"isolation_level": "AUTOCOMMIT"})
        return engine

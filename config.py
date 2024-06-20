import os

basedir = os.path.abspath(os.path.dirname(__file__))

class Config(object):
    '''Environment variabbles'''

    PROJECT_NAME = 'flaskai'
    APP_NAME = 'server'
    APP_RUNTIME = '3.11.3'
    APP_DESCRIPTION = 'simple app for sentiment analysis'

    APP_ENV = os.environ.get('APP_ENV') or 'dev'
    FLASK_ENV = os.environ.get('APP_ENV') or 'dev'
    DEBUG = os.environ.get('DEBUG') or True

    LOG_TO_STDOUT = os.environ.get('LOG_TO_STDOUT') or True
    SECRET_KEY = os.environ.get('SECRET_KEY') or b's0meSuperK3yHere!'

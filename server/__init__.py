import logging
import os
from logging.handlers import RotatingFileHandler
from flask import Flask
from config import Config


def configure_logging(app):
    # Configure the app's logging.
    if app.debug or app.testing:
        if app.config.get('LOG_TO_STDOUT'):
            stream_handler = logging.StreamHandler()
            stream_handler.setLevel(logging.INFO)
            app.logger.addHandler(stream_handler)
        else:
            log_dir = 'logs'
            os.makedirs(log_dir, exist_ok=True)
            file_handler = RotatingFileHandler(
                os.path.join(log_dir, f'{app.config["APP_NAME"]}.log'),
                maxBytes=20480,
                backupCount=20,
            )
            file_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]"
                )
            )
            file_handler.setLevel(logging.INFO)
            app.logger.addHandler(file_handler)
        app.logger.setLevel(logging.INFO)
        app.logger.info(f'{app.config["APP_NAME"]} startup')

def create_app():
    # Construct the core application.
    app = Flask(__name__, instance_relative_config=False)
    app.config.from_object(Config)

    with app.app_context():
        # Register API blueprint
        from server.main_bp import main_bp
        app.register_blueprint(main_bp, url_prefix='/')

        # Configure logging
        configure_logging(app)

    return app

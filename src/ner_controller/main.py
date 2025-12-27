"""Application entrypoint."""

from ner_controller.application.application_factory import ApplicationFactory
from ner_controller.configs.app_config import AppConfig

app = ApplicationFactory(AppConfig()).create_app()

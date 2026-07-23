from alembic import command
from alembic.config import Config

from app.config import PROJECT_ROOT, get_settings
from app.db.base import Base
from app.db.session import engine


def create_tables() -> None:
    Base.metadata.create_all(bind=engine)


def migrate_database() -> None:
    config = Config(str(PROJECT_ROOT / "alembic.ini"))
    config.set_main_option("sqlalchemy.url", get_settings().database_url)
    command.upgrade(config, "head")

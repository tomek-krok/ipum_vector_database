from sqlalchemy.engine import URL
from sqlalchemy import create_engine

def create_db_url():
    db_url = URL.create(
    drivername="postgresql+psycopg",
    username="postgres",
    password="password",
    host="localhost",
    port=5555,
    database="similarity_search_service_db"
    )
    return db_url

def create_db_engine(db_url):
    db_engine = create_engine(db_url)
    return db_engine

def get_engine():
    db_url = create_db_url()
    db_engine = create_db_engine(db_url)
    return db_engine

from db_table import Base, Images

# db_url = db_con.create_db_url()
# engine = db_con.create_db_engine(db_url)
engine = get_engine()
Base.metadata.create_all(engine)


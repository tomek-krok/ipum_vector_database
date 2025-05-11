import db_con
from db_table import Base, Images

# db_url = db_con.create_db_url()
# engine = db_con.create_db_engine(db_url)
engine = db_con.get_engine()
Base.metadata.drop_all(engine)
Base.metadata.create_all(engine)
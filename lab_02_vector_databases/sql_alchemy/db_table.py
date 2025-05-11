from pgvector.sqlalchemy import Vector
from sqlalchemy import Integer, String, Float, Boolean
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


# Create the base class for the table definition
class Base(DeclarativeBase):
    __abstract__ = True


# Create the table definition
class Images(Base):
    __tablename__ = "images"
    VECTOR_LENGTH = 512
    
    # primary key
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    # image path - we will use it to store the path to the image file, after similarity search we can use it to retrieve the image and display it
    image_path: Mapped[str] = mapped_column(String(256))
    # image embedding - we will store the image embedding in this column, the image embedding is a list of 512 floats this is the output of the sentence transformer model
    image_embedding: Mapped[list[float]] = mapped_column(Vector(VECTOR_LENGTH))

class Games(Base):

    __tablename__ = "games"
    __table_args__ = {'extend_existing': True}
    
    # the vector size produced by the model taken from documentation https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v2
    
    VECTOR_LENGTH = 512 
        
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(256))
    description: Mapped[str] = mapped_column(String(4096))
    windows: Mapped[bool] = mapped_column(Boolean)
    linux: Mapped[bool] = mapped_column(Boolean)
    mac: Mapped[bool] = mapped_column(Boolean)
    price: Mapped[float] = mapped_column(Float)
    game_description_embedding: Mapped[list[float]] = mapped_column(Vector(VECTOR_LENGTH))

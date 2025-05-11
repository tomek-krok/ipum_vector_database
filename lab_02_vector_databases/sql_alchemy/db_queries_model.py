from tqdm import tqdm
from sqlalchemy import Engine

from db_con import get_engine
from db_table import Games
from load_sentence_model import generate_embeddings

from sqlalchemy.orm import Session 
from download_game import load_dataset_game

from typing import Optional
from sqlalchemy import select

engine = get_engine()
dataset = load_dataset_game()



def insert_games(engine, dataset):
    with tqdm(total=len(dataset)) as pbar:
        for i, game in enumerate(dataset):
            game_description = game["About the game"] or ""
            game_embedding = generate_embeddings(game_description)
            name = game["Name"]
            windows = game["Windows"]
            linux = game["Linux"]
            mac =  game["Mac"]
            price = game["Price"]
            if name and windows and linux and mac and price and game_description:
                MyGame = Games(name=game["Name"], 
                                description=game_description[0:4096],
                                windows=game["Windows"], 
                                linux=game["Linux"], 
                                mac=game["Mac"], 
                                price=game["Price"], 
                                game_description_embedding=game_embedding
                            )   
                with Session(engine) as session:
                    if isinstance(MyGame, Games):
                        session.add(MyGame)
                        session.commit()
                    pbar.update(1)
        

insert_games(engine, dataset)

def find_game( engine: Engine, game_description: str, 
                windows: Optional[bool] = None, 
                linux: Optional[bool] = None,
                mac: Optional[bool] = None,
                price: Optional[int] = None
                ):
    
    game_embedding = generate_embeddings(game_description)

    with Session(engine) as session:
        query = (
            select(Games)
            .order_by(Games.game_description_embedding.cosine_distance(game_embedding))
        )
        
        if price:
            query = query.filter(Games.price <= price)
        if windows:
            query = query.filter(Games.windows == True)
        if linux:
            query = query.filter(Games.linux == True)
        if mac:
            query = query.filter(Games.mac == True)
        
        result = session.execute(query, execution_options={"prebuffer_rows": True})
        game = result.scalars().first()
        
        return game
    
# game = find_game(engine, "This is a game about a hero who saves the world", price=10)
# print(f"Game: {game.name}")
# print(f"Description: {game.description}")

# game = find_game(engine, game_description="Home decorating", price=20)
# print(f"Game: {game.name}")
# print(f"Description: {game.description}")

game = find_game(engine, game_description="Home decorating", mac=True, price=5)
print(f"Game: {game.name}")
print(f"Description: {game.description}")
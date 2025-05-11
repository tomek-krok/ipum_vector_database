from datasets import load_dataset

def load_dataset_game():
    dataset = load_dataset("FronkonGames/steam-games-dataset")

# get columns names and types
    columns = dataset["train"].features
    print(columns)

    columns_to_keep = ["Name", "Windows", "Linux", "Mac", "About the game", "Supported languages", "Price"]

    N = 1000
    dataset = dataset["train"].select_columns(columns_to_keep).select(range(N))
    return dataset


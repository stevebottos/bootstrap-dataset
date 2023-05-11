import requests
import typer
from pymilvus import Collection, connections, utility
from rich.console import Console
import numpy as np

import server.constants as constants

console = Console()
app = typer.Typer()
URL = "http://127.0.0.1"
PORT = 8000


def print(*args):
    console.print(*args)


@app.command(help="Drops the current 'laion' collection and associated indices.")
def drop_laion():
    requests.post(f"{URL}:{PORT}/drop_laion")


@app.command(help="Query the database with an image to obtain matches.")
def query_laion(
    url: str = typer.Argument(
        ..., help="Path to the target image you wish to query with."
    ),
    similarity_threshold: float = typer.Option(
        0.75,
        help="The minimum similarity that an image must share with the query to be returned as a match.",
    ),
    max_results: int = typer.Option(
        1000,
        help="The maximum number of results to return.",
    ),
):
    res = requests.post(
        f"{URL}:{PORT}/query_laion",
        json={
            "url": url,
            "similarity_threshold": similarity_threshold,
            "max_results": max_results,
        },
    )

    data = res.json()
    urls = []
    similarities = []
    for data in res.json().values():
        similarities.extend(data["similarity"])
        urls.extend(data["url"])

    similarities = np.round(similarities, 3)

    data = sorted(zip(similarities, urls))
    print(*data)


@app.command(help="Loads all laion indices into milvus")
def download_laion(
    index: int = typer.Argument(..., help="The laion shard to retrieve.")
):
    requests.post(f"{URL}:{PORT}/download_laion", json={"index": index})


@app.command(help="Adds (and indexes) a shard of laion to the database")
def init_laion(
    batchsize: int = typer.Option(
        10000, help="The batchsize to update the database in. This eases RAM usage."
    ),
    limit: int = typer.Option(
        0,
        help="Useful for debugging. If > 0 then only n-embeddings will be used where n = limit",
    ),
):
    requests.post(
        f"{URL}:{PORT}/init_laion",
        json={"batchsize": batchsize, "limit": limit},
    )


# TODO: Make endpoint, just for dev rn
@app.command(help="(Dev) Check the current collection.")
def check_laion():
    connections.connect(host=constants.HOST, port=constants.PORT)
    collections = utility.list_collections()
    if len(collections):
        collection = Collection(constants.COLLECTION_NAME)
        print(collection.num_entities)
        print(collection.describe())


if __name__ == "__main__":
    app()

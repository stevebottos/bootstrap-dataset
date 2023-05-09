import math
import os
from io import BytesIO

import clip
import cv2
import numpy as np
import pandas as pd
import requests
import torch
import torchvision.transforms.functional as F
import typer
from PIL import Image
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)
from rich.console import Console
from rich.status import Status
from tqdm import tqdm

import constants
from util import inference

console = Console()
app = typer.Typer()

TTL = 30  # Set higher for more time to live


def print(*args):
    console.print(*args)


def search(collection, vector_field, search_vectors, urls):
    search_param = {
        "data": search_vectors,
        "anns_field": vector_field,
        "param": {
            "metric_type": constants.METRIC_TYPE,
            "params": {"nprobe": constants.NPROBE},
        },
        "limit": constants.TOPK,
        "expr": "id_field >= 0",
    }
    results = collection.search(**search_param)
    for i, result in enumerate(results):
        print("\nSearch result for {}th vector: ".format(i))
        for j, res in enumerate(result):
            print("Top {}: {} ({})".format(j, urls[res.id], res.distance))

    return results


@app.command(help="Query the database with an image to obtain matches.")
def query(
    image_path: str = typer.Argument(
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
    connections.connect(host=constants.HOST, port=constants.PORT)
    collection = Collection(constants.COLLECTION_NAME)
    urls = pd.read_parquet("laion/0/metadata_0.parquet")["url"]
    image_features = inference.inference(image_path)
    _ = search(collection, constants.VECTOR_FIELD_NAME, image_features, urls)


@app.command(
    help="Creates an empty collection with happy presets in the db. Ensure that the db is running first."
)
def create_empty_laion_collection():

    connections.connect(host=constants.HOST, port=constants.PORT)

    field1 = FieldSchema(
        name=constants.ID_FIELD_NAME,
        dtype=DataType.INT64,
        description="int64",
        is_primary=True,
    )
    field2 = FieldSchema(
        name=constants.SHARD_FIELD_NAME,
        dtype=DataType.INT64,
        description="int64",
        is_primary=False,
    )
    field3 = FieldSchema(
        name=constants.VECTOR_FIELD_NAME,
        dtype=DataType.FLOAT_VECTOR,
        description="float vector",
        dim=constants.DIM,
        is_primary=False,
    )
    schema = CollectionSchema(
        fields=[field1, field2, field3], description="collection description"
    )
    collection = Collection(
        name=constants.COLLECTION_NAME,
        data=None,
        schema=schema,
        properties={"collection.ttl.seconds": TTL},
    )

    index_param = {
        "index_type": constants.INDEX_TYPE,
        "params": {"nlist": constants.NLIST},
        "metric_type": constants.METRIC_TYPE,
    }
    collection.set_properties(properties={"collection.ttl.seconds": 0})
    collection.create_index(constants.VECTOR_FIELD_NAME, index_param)
    print(f"Collection {constants.COLLECTION_NAME} created.")


@app.command()
def get_laion_shard(
    shard_index: int = typer.Argument(..., help="The laion shard to retrieve.")
):
    outfolder = f"{constants.DATAFOLDER}/{shard_index}"
    os.mkdir(outfolder)

    for url in [
        constants.METADATA.format(idx=shard_index),
        constants.INDEX.format(idx=shard_index),
    ]:
        response = requests.get(url, stream=True)
        total_size_in_bytes = int(response.headers.get("content-length", 0))
        block_size = 1024
        progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)
        fname = os.path.basename(url)
        print(f"Downloading from {url} to {outfolder}/{fname}")

        with open(f"{outfolder}/{fname}", "wb") as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()
        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            print("ERROR, something went wrong")


@app.command(help="Adds (and indexes) a shard of laion to the database")
def batch_update_database(
    shard_index: int = typer.Argument(..., help="The laion shard to retrieve."),
    batchsize: int = typer.Option(
        10000, help="The batchsize to update the database in. This eases RAM usage."
    ),
    limit: int = typer.Option(
        0,
        help="Useful for debugging. If > 0 then only n-embeddings will be used where n = limit",
    ),
):

    embeddings = np.load(
        f"{constants.DATAFOLDER}/{shard_index}/img_emb_{shard_index}.npy"
    )
    if limit:
        embeddings = embeddings[:limit]

    indices = [i for i in range(len(embeddings))]
    shards_col = [shard_index] * len(embeddings)

    n_chunks = math.ceil(len(embeddings) / batchsize)

    connections.connect(host=constants.HOST, port=constants.PORT)

    if not utility.has_collection(constants.COLLECTION_NAME):
        print(
            "No collection id exists yet. Run 'python commands.py create-empty-laion-collection'"
        )
        return
    status = Status(spinner="monkey", status="")
    status.start()
    collection = Collection(constants.COLLECTION_NAME)
    for i in range(n_chunks):
        status.update(f"Ingesting chunk {i+1} of {n_chunks}")
        start = i * batchsize
        end = min((i + 1) * batchsize, len(embeddings))

        data = [indices[start:end], shards_col[start:end], embeddings[start:end]]
        collection.insert(data)
        collection.flush()

    status.console.print("All data ingested successfully.")
    status.stop()


@app.command(
    help="Loads the database into RAM for fast querying. By default, this step happens automatically after ingesting. This must be done before querying, and only needs to be done once."
)
def load():
    status = Status(spinner="monkey", status="Loading...")
    status.start()
    connections.connect(host=constants.HOST, port=constants.PORT)
    collection = Collection(constants.COLLECTION_NAME)
    collection.load()
    status.console.print(
        f"All loaded successfully. Hosting {collection.num_entities} embeddings, ready for query."
    )
    status.stop()


@app.command(help="Wipes the current collection and associated indices.")
def wipe_collection():
    connections.connect(host=constants.HOST, port=constants.PORT)
    collection = Collection(constants.COLLECTION_NAME)
    collection.drop()


if __name__ == "__main__":
    app()

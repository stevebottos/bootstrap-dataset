import math
import os
import glob

import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI
from pydantic import BaseModel
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)
from rich.console import Console
from tqdm import tqdm

from . import constants
from .util import inference

app = FastAPI()
console = Console()
TTL = 0


class Query(BaseModel):
    url: str
    similarity_threshold: float = 0.75
    max_results: int = 1000


class LaionIndex(BaseModel):
    index: int


class Update(BaseModel):
    index: int
    batchsize: int
    limit: int = 0


def print(*args):
    console.print(*args)


def _search(collection, vector_field, search_vectors, urls):
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


@app.post("/init_collection")
def init_collection():
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


@app.post("/load_collection")
def load_collection():
    connections.connect(host=constants.HOST, port=constants.PORT)
    collection = Collection(constants.COLLECTION_NAME)
    collection.load()


@app.post("/update_laion")
def update_laion(update: Update):
    index = update.index
    batchsize = update.batchsize
    limit = update.limit

    embeddings = np.load(f"{constants.DATAFOLDER}/{index}/img_emb_{index}.npy")
    if limit:
        embeddings = embeddings[:limit]

    indices = [i for i in range(len(embeddings))]
    shards_col = [index] * len(embeddings)

    n_chunks = math.ceil(len(embeddings) / batchsize)

    connections.connect(host=constants.HOST, port=constants.PORT)

    if not utility.has_collection(constants.COLLECTION_NAME):
        print(
            "No collection id exists yet. Run 'python commands.py create-empty-laion-collection'"
        )
        return

    collection = Collection(constants.COLLECTION_NAME)
    for i in range(n_chunks):
        start = i * batchsize
        end = min((i + 1) * batchsize, len(embeddings))

        data = [indices[start:end], shards_col[start:end], embeddings[start:end]]
        collection.insert(data)
        collection.flush()

        print(f"Loaded batch {i+1} of {n_chunks}")


@app.post("/query_collection")
def query_collection(query: Query):
    connections.connect(host=constants.HOST, port=constants.PORT)
    collection = Collection(constants.COLLECTION_NAME)
    urls = pd.read_parquet("server/laion/0/metadata_0.parquet")["url"]
    image_features, err = inference.inference(query.url)
    if err:
        console.print(f":x: Failed with:\n{err}.")

    _ = _search(collection, constants.VECTOR_FIELD_NAME, image_features, urls)


@app.post("/drop_collection")
def drop_collection():
    connections.connect(host=constants.HOST, port=constants.PORT)
    collection = Collection(constants.COLLECTION_NAME)
    collection.drop()


@app.post("/download_laion")
def download_laion(index: LaionIndex):
    shard_index = index.index

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

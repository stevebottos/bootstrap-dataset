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
    batchsize: int = 10000
    limit: int = 0


def print(*args):
    console.print(*args)


def _search(collection, vector_field, search_vectors):
    search_param = {
        "data": search_vectors,
        "anns_field": vector_field,
        "param": {
            "metric_type": constants.METRIC_TYPE,
            "params": {"nprobe": constants.NPROBE},
        },
        "limit": constants.TOPK,
        "output_fields": [constants.SHARD_FIELD_NAME],
        "expr": "id_field >= 0",
    }
    results = collection.search(**search_param)
    return results


def _init_laion():
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
        fields=[field1, field2, field3],
        description="collection description",
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


@app.post("/init_laion")
def init_laion(update: Update):
    _init_laion()

    batchsize = update.batchsize
    limit = update.limit

    embedding_files = glob.glob(f"{constants.DATAFOLDER}/**/*.npy")
    n_shards = len(embedding_files)

    embeddings = []
    for j, embedding_file in enumerate(embedding_files):
        index = int(embedding_file.split("/")[2])

        embeddings = np.load(embedding_file)

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

            data = [
                indices[start:end],
                shards_col[start:end],
                embeddings[start:end],
            ]
            collection.insert(data)
            collection.flush()

            print(f"Ingested batch {i+1} of {n_chunks}, shard {j+1} of {n_shards}")
            break

        if limit:
            break

    print(f"Loading {collection.num_entities} entries into memory.")
    collection.load()
    print(f"{collection.num_entities} entries loaded into memory.")


@app.post("/query_laion")
def query_laion(query: Query):
    url = query.url
    threshold = query.similarity_threshold
    max_results = query.max_results

    connections.connect(host=constants.HOST, port=constants.PORT)
    collection = Collection(constants.COLLECTION_NAME)

    image_features, err = inference.inference(url)
    if err:
        console.print(f":x: Failed with:\n{err}.")

    results = _search(collection, constants.VECTOR_FIELD_NAME, image_features)[0]

    from collections import defaultdict

    data = defaultdict(lambda: defaultdict(list))

    for r in results:
        data[r.entity.get("laion_shard")]["indices"].append(r.id)
        data[r.entity.get("laion_shard")]["similarity"].append(r.distance)

    for shard in data:
        urls = pd.read_parquet(
            f"{constants.DATAFOLDER}/{shard}/metadata_{shard}.parquet"
        )["url"]
        indices = data[shard]["indices"]
        data[shard]["url"] = [urls[i] for i in indices]
        del data[shard]["indices"]

    return data


@app.post("/drop_laion")
def drop_laion():
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

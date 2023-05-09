# for milvus server
HOST = "localhost"
PORT = 19530
COLLECTION_NAME = "laion"
ID_FIELD_NAME = "id_field"
VECTOR_FIELD_NAME = "clip_embedding"
SHARD_FIELD_NAME = "laion_shard"
DIM = 512
INDEX_TYPE = "IVF_SQ8"  # "IVF_FLAT" <- slower query but slightly more accurate...
METRIC_TYPE = "IP"
NLIST = 1024
TOPK = 250
NPROBE = 16

# for data retrieval
BASE = "https://deploy.laion.ai/8f83b608504d46bb81708ec86e912220/embeddings"
METADATA = BASE + "/metadata/metadata_{idx}.parquet"
INDEX = BASE + "/img_emb/img_emb_{idx}.npy"
DATAFOLDER = "laion"

import argparse, json
from datetime import datetime
from tqdm import tqdm
from db import get_collection
from embed import set_encoder, encode_many
from config import EMBEDDING_CONFIGS


def load_chunks(path: str):
    """
    Input file format: JSONL, each line is a dict with fields:
    { "_id", "parent_id", "chunk_index", "task", "source", "text", "meta": {...} }
    """
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def ingest(model_key: str, input_file: str, batch_size: int = 128):
    if model_key not in EMBEDDING_CONFIGS:
        raise ValueError(f"Unknown model_key={model_key}. Must be one of {list(EMBEDDING_CONFIGS.keys())}")

    cfg = EMBEDDING_CONFIGS[model_key]
    print(f"[INFO] Ingesting with model={cfg['hf_model']} into collection={cfg['collection']}")

    set_encoder(cfg["hf_model"])

    col = get_collection("rag_med", cfg["collection"])

    buf = []
    total = 0
    for item in tqdm(load_chunks(input_file), desc="Reading"):
        buf.append(item)
        if len(buf) >= batch_size:
            texts = [c["text"] for c in buf]
            vecs = encode_many(texts)
            docs = []
            for c, v in zip(buf, vecs):
                docs.append({
                    "_id": c["_id"],
                    "parent_id": c["parent_id"],
                    "chunk_index": c["chunk_index"],
                    "task": c.get("task"),
                    "source": c.get("source"),
                    "text": c["text"],
                    "meta": c.get("meta", {}),
                    "vector": v.tolist(),
                    "model": cfg["hf_model"],
                    "created_at": datetime.utcnow(),
                })
            if docs:
                col.insert_many(docs, ordered=False)
                total += len(docs)
            buf.clear()

    # Flush
    if buf:
        texts = [c["text"] for c in buf]
        vecs = encode_many(texts)
        docs = []
        for c, v in zip(buf, vecs):
            docs.append({
                "_id": c["_id"],
                "parent_id": c["parent_id"],
                "chunk_index": c["chunk_index"],
                "task": c.get("task"),
                "source": c.get("source"),
                "text": c["text"],
                "meta": c.get("meta", {}),
                "vector": v.tolist(),
                "model": cfg["hf_model"],
                "created_at": datetime.utcnow(),
            })
        if docs:
            col.insert_many(docs, ordered=False)
            total += len(docs)

    print(f"[DONE] Inserted {total} docs into {cfg['collection']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_key", required=True, help="gemma | medembed | spubmed")
    parser.add_argument("--input", required=True, help="Path to JSONL chunk file")
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    ingest(args.model_key, args.input, batch_size=args.batch_size)

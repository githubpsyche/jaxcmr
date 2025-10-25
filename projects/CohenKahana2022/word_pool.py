# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: .jupytext-sync-ipynb//ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: jaxcmr
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Build semantic similarity (Sentence-Transformers) for USF words
#
# This percent-formatted notebook:
# 1) loads `usf_words.txt` (one word per line),
# 2) encodes words with a Sentence-Transformers model,
# 3) computes a cosine similarity matrix, and
# 4) saves it to: `florida_nouns-{modelname}.npy` in a target directory.
#
# Notes:
# - We set `normalize_embeddings=True` so cosine similarity is simple `emb @ emb.T`.
# - Result is `float32`, symmetric, with 1.0 on the diagonal.

# %%
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

# %% [markdown]
# ## Config
# Edit these as needed; the file must be one word per line.

# %%
WORDS_PATH = Path("data/raw/CohenKahana2022/usf_words.txt")     # input list (one word per line)
OUT_DIR = Path("data/")                      # where to save the .npy file
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"  # try: all-mpnet-base-v2 for higher quality
BATCH_SIZE = 256

# %% [markdown]
# ## Load vocabulary

# %%
words = [w.strip() for w in WORDS_PATH.read_text().splitlines() if w.strip()]
print(f"Loaded {len(words)} words from {WORDS_PATH}")

# %% [markdown]
# ## Encode with Sentence-Transformers
# We normalize embeddings so dot product equals cosine similarity.

# %%
model = SentenceTransformer(MODEL_NAME)
emb = model.encode(
    words,
    batch_size=BATCH_SIZE,
    convert_to_numpy=True,
    normalize_embeddings=True,
    show_progress_bar=True,
)
emb = emb.astype(np.float32)
print("Embeddings:", emb.shape, emb.dtype)

# %% [markdown]
# ## Cosine similarity matrix

# %%
sim = emb @ emb.T  # cosine, since normalized
sim = sim.astype(np.float32)
print("Similarity matrix:", sim.shape, sim.dtype, f"range=({sim.min():.3f},{sim.max():.3f})")

# %% [markdown]
# ## Save as `florida_nouns-{modelname}.npy`

# %%
OUT_DIR.mkdir(parents=True, exist_ok=True)
model_tag = MODEL_NAME.split("/")[-1]
out_path = OUT_DIR / f"florida_nouns-{model_tag}.npy"
np.save(out_path, sim)
print("Saved:", out_path)

# %% [markdown]
# ## (Optional) Save embeddings too
# Uncomment if you also want the raw embeddings for later use.

# %%
# np.save(OUT_DIR / f"florida_nouns-embeddings-{model_tag}.npy", emb)
# print("Saved embeddings:", OUT_DIR / f"florida_nouns-embeddings-{model_tag}.npy")

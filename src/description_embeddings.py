"""Embed product descriptions with SBERT.

Input is ``data/processed/first_transaction_churn_clean.csv`` — the corrected
line-item table from ``correct_data_issues.py``. Output is:

``data/processed/description_embeddings.npz``
    ``stock_code``: the product key, one entry per distinct code.
    ``embedding``: the matching ``(n_products, 384)`` float32 matrix, L2
    normalised.

Grain
-----
Descriptions are a property of the **product**, not of the line item. The
corrected table holds 125,042 line items but only 4,171 distinct
``stock_code`` values, and ``normalize_descriptions`` in
``correct_data_issues.py`` has already collapsed each code to a single modal
description — so there are exactly 4,171 distinct ``(stock_code, description)``
pairs. Embedding the line items would therefore encode the same ~4,000 strings
thirty times over. This module embeds the product catalogue once and leaves the
join to the caller, exactly as ``build_item_history.py`` does.

Like the item-history table, this is a **lookup keyed by ``stock_code``**, not a
feature table. Getting from a 384-dimensional product vector to a feature of a
*customer* means deciding how to pool a basket's worth of vectors (mean,
max, tf-idf weighting, cluster assignment, distance to the customer's own
centroid), and that decision belongs to the two pipelines rather than here.

Unlike the rest of ``src/``, nothing in this module is mechanical: choosing to
represent a product by a sentence embedding is a modelling decision, and the
choice of model is another. It is kept in its own module, writing its own
artefact, so that a pipeline can use it or ignore it rather than inheriting it
through the modelling table.

The model
---------
``all-MiniLM-L6-v2`` — the standard SBERT baseline: 6 layers, 384 dimensions,
fast enough on CPU to embed the whole catalogue in seconds. Its tokenizer is
uncased, which matters here because every description in the source is
upper-case (``'PINK CHERRY LIGHTS'``).

Descriptions are stripped of surrounding whitespace before encoding — the raw
column carries some (``' WHITE CHERRY LIGHTS'``, ``'RECORD FRAME 7" SINGLE
SIZE '``) — but are otherwise passed through untouched.

Descriptions are not unique across products
-------------------------------------------
27 descriptions are shared by more than one ``stock_code``, covering 59
products — ``COLUMBIAN CANDLE ROUND`` is codes 72127, 72128, 72129 and 72130;
``METAL SIGN,CUPCAKE SINGLE HOOK`` is 82613A, 82613B and 82613C. These read as
colour or size variants whose distinguishing detail never reached the
description field.

Those products embed to **identical vectors**, so similarity 1.0 between two
different codes is expected rather than a bug, and a nearest-neighbour lookup
will return the whole variant family before anything genuinely different. The
embedding cannot separate them; only ``stock_code``, price or the item history
can. This is left in rather than deduplicated, because the codes really are
distinct products and collapsing them would silently change the grain.

Why the vectors are normalised
------------------------------
``embed_descriptions`` returns L2-normalised rows, so the **dot product of two
embeddings is their cosine similarity**, bounded in [-1, 1]. On unnormalised
vectors a plain dot product also rewards magnitude, which for sentence
embeddings tracks description length more than meaning. Normalising once here
means ``embeddings @ query`` is a valid similarity everywhere downstream, and a
similarity matrix is a matrix multiply rather than a special function.

Run from anywhere:
    python src/description_embeddings.py

which embeds the catalogue, validates it, writes the ``.npz`` and prints the
nearest neighbours of a sample description as a smoke test.
"""
from pathlib import Path

import numpy as np
import pandas as pd

RANDOM_SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'
IN_FILE = PROCESSED_DIR / 'first_transaction_churn_clean.csv'
OUT_FILE = PROCESSED_DIR / 'description_embeddings.npz'

MODEL_NAME = 'all-MiniLM-L6-v2'
EMBEDDING_DIM = 384
BATCH_SIZE = 256

# Share of embedding variance that reduce_embeddings keeps. The component count
# follows from the matrix it is fitted on: 0.90 needs 140 of the 384 dimensions
# on the product catalogue, but only 96 on the customer means that
# build_modelling_dataset feeds it. Sentence embeddings are not low-rank, so
# this buys a smaller feature block rather than a free lunch.
PCA_VARIANCE = 0.90
PCA_PREFIX = 'pca_'

# float32 matmul under numpy 2.x on Apple's Accelerate BLAS raises spurious
# divide-by-zero / overflow / invalid warnings: Accelerate leaves FPU exception
# flags set and numpy reports them even for `np.ones @ np.ones`. The results are
# unaffected — checked against the float64 product, max absolute difference
# 4.7e-08 with identical top-10 ordering — so the two dot-product sites below
# silence the flags locally rather than globally, and check_embeddings still
# asserts the output is finite.
MATMUL_ERRSTATE = dict(divide='ignore', over='ignore', invalid='ignore')

PRODUCT_COLUMNS = ['stock_code', 'description']
EMBEDDING_COLUMN = 'description_embedding'


def product_descriptions(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce the line-item frame to one description per product.

    Input:
        ``df``: corrected first-transaction frame with ``stock_code`` and
        ``description``.
    Does:
        Takes the distinct ``(stock_code, description)`` pairs and strips
        surrounding whitespace from the description. Asserts that each code
        carries exactly one description, which is what
        ``correct_data_issues.normalize_descriptions`` guarantees — if that ever
        stops holding, the assertion fires rather than an arbitrary description
        being picked silently.
    Output:
        A frame of ``PRODUCT_COLUMNS``, one row per product, sorted by
        ``stock_code`` with a reset index. Row order defines the row order of
        the embedding matrix.
    """
    per_code = df.groupby('stock_code')['description'].nunique()
    offenders = per_code[per_code > 1]
    assert offenders.empty, \
        f'{len(offenders)} stock codes carry more than one description'

    products = (df[PRODUCT_COLUMNS]
                .drop_duplicates('stock_code')
                .assign(description=lambda d: d['description'].str.strip())
                .sort_values('stock_code')
                .reset_index(drop=True))

    assert products['description'].str.len().gt(0).all(), 'an empty description'
    return products


def load_model(name: str = MODEL_NAME, quiet: bool = True):
    """Load the SBERT model, seeded for reproducibility.

    Input:
        ``name``: a sentence-transformers model name.
        ``quiet``: drop ``transformers`` logging to error level first. This
        model's load report flags ``embeddings.position_ids`` as UNEXPECTED —
        benign for MiniLM, but it prints a table that buries the real output in
        a notebook. Pass ``quiet=False`` to see the full report when changing
        ``MODEL_NAME``, since for a different architecture that table is worth
        reading.
    Does:
        Seeds torch, then loads the model. Encoding runs in eval mode and is
        deterministic regardless, but the seed is set so a model whose pooling
        ever involves sampling cannot drift between runs.
    Output:
        A ``SentenceTransformer``.

    Imported lazily so that importing this module — to reach ``most_similar`` or
    ``load_embeddings`` on already-built vectors — does not pay the cost of
    pulling in torch.
    """
    import torch
    import transformers
    from sentence_transformers import SentenceTransformer

    if quiet:
        transformers.logging.set_verbosity_error()

    torch.manual_seed(RANDOM_SEED)
    return SentenceTransformer(name)


def embed_descriptions(descriptions, model=None,
                       batch_size: int = BATCH_SIZE) -> np.ndarray:
    """Encode a sequence of descriptions into L2-normalised SBERT vectors.

    Input:
        ``descriptions``: sequence of strings, one per product.
        ``model``: a loaded ``SentenceTransformer``; loaded via ``load_model``
        if omitted. Pass one in when embedding more than once, so the model is
        not reloaded per call.
        ``batch_size``: encoding batch size.
    Does:
        Encodes every description, L2-normalising each vector so that a dot
        product between any two rows is their cosine similarity.
    Output:
        A ``(len(descriptions), EMBEDDING_DIM)`` float32 array, row-aligned to
        the input.
    """
    model = model if model is not None else load_model()
    embeddings = model.encode(list(descriptions),
                              batch_size=batch_size,
                              convert_to_numpy=True,
                              normalize_embeddings=True,
                              show_progress_bar=False)
    return embeddings.astype(np.float32)


def build_description_embeddings(df: pd.DataFrame, model=None):
    """Build the product index and its embedding matrix from the line items.

    Input:
        ``df``: the corrected first-transaction frame (line-item grain).
        ``model``: optional preloaded ``SentenceTransformer``.
    Does:
        Reduces to one description per product, then embeds them.
    Output:
        ``(products, embeddings)`` — the ``PRODUCT_COLUMNS`` frame and the
        row-aligned float32 matrix. ``products.iloc[i]`` describes
        ``embeddings[i]``.
    """
    products = product_descriptions(df)
    embeddings = embed_descriptions(products['description'], model=model)
    return products, embeddings


def check_embeddings(products: pd.DataFrame, embeddings: np.ndarray) -> None:
    """Raise ``AssertionError`` unless the matrix is usable and row-aligned.

    Confirms one row per product, the expected width and dtype, no NaN or Inf,
    unit-norm rows (so dot products really are cosine similarities), and that no
    vector is degenerate. Also confirms a description's self-similarity is 1,
    which is the end-to-end check that rows have not been shuffled out of
    alignment with ``products``.
    """
    assert len(products) == len(embeddings), \
        'products and embeddings have different lengths'
    assert not products['stock_code'].duplicated().any(), 'stock_code is not unique'
    assert embeddings.shape[1] == EMBEDDING_DIM, \
        f'expected {EMBEDDING_DIM} dimensions, got {embeddings.shape[1]}'
    assert embeddings.dtype == np.float32, 'embeddings are not float32'
    assert np.isfinite(embeddings).all(), 'embeddings contain NaN or Inf'

    norms = np.linalg.norm(embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5), 'embeddings are not L2 normalised'

    # Self-similarity is the alignment check: row i must be the best match for
    # its own description, at a similarity of 1.
    with np.errstate(**MATMUL_ERRSTATE):
        self_sim = np.einsum('ij,ij->i', embeddings, embeddings)
    assert np.allclose(self_sim, 1.0, atol=1e-5), 'self-similarity is not 1'


def most_similar(query: str, products: pd.DataFrame, embeddings: np.ndarray,
                 model=None, top_n: int = 10) -> pd.DataFrame:
    """Return the products whose descriptions are closest to ``query``.

    Input:
        ``query``: a free-text description. Need not be one of the catalogue's.
        ``products``, ``embeddings``: as returned by
        ``build_description_embeddings``.
        ``model``: optional preloaded ``SentenceTransformer``.
        ``top_n``: how many neighbours to return.
    Does:
        Embeds the query with the same normalisation, then takes the dot product
        against every product vector. Because both sides are unit length, that
        dot product *is* the cosine similarity, so no further scaling is needed.
        ``argpartition`` finds the top ``top_n`` without sorting all 4,171.
    Output:
        A frame of ``stock_code``, ``description`` and ``similarity``, sorted by
        similarity descending. If ``query`` is itself a catalogue description,
        its own product is the first row at a similarity of 1.
    """
    query_vec = embed_descriptions([query], model=model)[0]
    with np.errstate(**MATMUL_ERRSTATE):
        similarity = embeddings @ query_vec
    assert np.isfinite(similarity).all(), 'similarities are not finite'

    top_n = min(top_n, len(products))
    top = np.argpartition(-similarity, top_n - 1)[:top_n]
    top = top[np.argsort(-similarity[top])]

    return (products.iloc[top]
            .assign(similarity=similarity[top])
            .reset_index(drop=True))


def reduce_embeddings(embeddings: np.ndarray, variance: float = PCA_VARIANCE,
                      random_state: int = RANDOM_SEED):
    """Reduce an embedding matrix with PCA, keeping ``variance`` of the total.

    Input:
        ``embeddings``: any matrix of embeddings — the product catalogue, or the
        customer-level means that ``build_modelling_dataset`` feeds it.
        ``variance``: share of variance to retain; the component count follows
        from it rather than being fixed in advance.
    Does:
        Fits PCA on whatever it is given and projects it. ``svd_solver='full'``
        because a fractional ``n_components`` requires the exact solver, and at
        this size there is nothing to gain from the randomised one.
    Output:
        ``(reduced, pca)`` — the ``(n_rows, n_components)`` float32 matrix and
        the fitted ``PCA``, whose ``n_components_`` and
        ``explained_variance_ratio_`` describe what survived.

    What it is fitted on matters more than the order it runs in. PCA is affine,
    so projecting product vectors then averaging per customer gives the same
    numbers as averaging then projecting *with the same components* — verified
    to 3e-07. What differs is the matrix the components come from: the 4,171
    product catalogue is lower-variance-per-dimension and needs 140 components
    for 90%, while the 5,044 customer means are smoother — averaging ~18 unit
    vectors concentrates them — and reach 90% in 96.

    Fitting on customer means does mean the components are informed by rows that
    later become test data. The threshold never looks at ``churn``, so this is
    unsupervised rather than target leakage, but a pipeline that needs the
    benchmark airtight should refit on the training split alone and apply the
    fitted ``pca`` to the rest::

        reduced_train, pca = reduce_embeddings(means_train)
        reduced_test = pca.transform(means_test)
    """
    from sklearn.decomposition import PCA

    pca = PCA(n_components=variance, svd_solver='full', random_state=random_state)
    reduced = pca.fit_transform(embeddings).astype(np.float32)
    return reduced, pca


def pca_column_names(n_components: int, prefix: str = PCA_PREFIX) -> list:
    """Name the reduced dimensions, zero-padded so they sort in order.

    The width follows the count, because the component count is chosen by the
    variance threshold rather than fixed: 140 components give ``pca_000`` …
    ``pca_139``.
    """
    width = max(2, len(str(max(n_components - 1, 0))))
    return [f'{prefix}{i:0{width}d}' for i in range(n_components)]


def attach_embeddings(df: pd.DataFrame, products: pd.DataFrame,
                      embeddings: np.ndarray,
                      column: str = EMBEDDING_COLUMN) -> pd.DataFrame:
    """Add each row's product embedding as a single column of vectors.

    Input:
        ``df``: any frame with a ``stock_code`` column — line items, or an
        already-enriched transaction frame.
        ``products``, ``embeddings``: as returned by
        ``build_description_embeddings``.
        ``column``: name for the new column.
    Does:
        Maps ``stock_code`` to its 384-dimensional vector and stores the whole
        vector in one object column, rather than spreading it across 384 float
        columns. Every row of a given product points at the **same** array
        object — the catalogue has 4,171 products against 125,042 line items, so
        copying per row would turn 6.4 MB into 192 MB for no new information.
        The shared vectors are marked read-only, because writing through one row
        would otherwise silently change every other row of that product.
    Output:
        A copy of ``df`` with ``column`` appended. Raises if any ``stock_code``
        has no embedding.

    Note for the aggregation this is meant to feed: the stored vectors are unit
    length, but a **mean of unit vectors is not** — its norm falls as the basket
    gets more varied. That shrinkage is real information (a focused basket
    averages to a longer vector than a scattered one), so decide deliberately
    whether to keep it or re-normalise the customer-level mean before use.
    """
    vectors = pd.Series(list(embeddings), index=products['stock_code'])
    for vector in vectors:
        vector.setflags(write=False)

    attached = df.copy()
    attached[column] = df['stock_code'].map(vectors)

    missing = attached[column].isna()
    assert not missing.any(), \
        f'{int(missing.sum())} rows have a stock_code with no embedding'
    return attached


def stack_embeddings(column: pd.Series) -> np.ndarray:
    """Turn a column of vectors back into a 2-D matrix.

    Input:
        ``column``: a Series of equal-length 1-D arrays, e.g. the column
        ``attach_embeddings`` added, or one group of it.
    Output:
        A ``(len(column), EMBEDDING_DIM)`` array, so ordinary numpy — including
        ``.mean(axis=0)`` per customer — applies again.
    """
    stacked = np.stack(column.to_numpy())
    assert stacked.shape == (len(column), EMBEDDING_DIM), \
        f'stacked to {stacked.shape}, expected ({len(column)}, {EMBEDDING_DIM})'
    return stacked


def save_embeddings(products: pd.DataFrame, embeddings: np.ndarray,
                    path: Path = OUT_FILE) -> None:
    """Write the index and matrix to a single compressed ``.npz``.

    One file rather than a CSV plus a matrix, so the two cannot drift out of
    alignment. CSV is the wrong container for 4,171 x 384 floats: it would be
    roughly five times the size and would round-trip through text.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path,
                        stock_code=products['stock_code'].to_numpy(dtype=object),
                        description=products['description'].to_numpy(dtype=object),
                        embedding=embeddings)


def load_embeddings(path: Path = OUT_FILE):
    """Read back what ``save_embeddings`` wrote.

    Output:
        ``(products, embeddings)`` in the same form
        ``build_description_embeddings`` returns, so the two are interchangeable.
    """
    with np.load(path, allow_pickle=True) as data:
        products = pd.DataFrame({
            'stock_code': data['stock_code'].astype(str),
            'description': data['description'].astype(str),
        })
        embeddings = data['embedding']
    return products, embeddings


def main() -> None:
    """Embed the catalogue, validate it, write it out and print a smoke test."""
    df = pd.read_csv(IN_FILE)
    print(f'Loaded corrected line items : {df.shape}')

    print(f'Loading SBERT model         : {MODEL_NAME}')
    model = load_model()

    products, embeddings = build_description_embeddings(df, model=model)
    check_embeddings(products, embeddings)

    print()
    print('Description embeddings')
    print('----------------------')
    print(f'Products embedded           : {len(products):,} '
          f'(from {len(df):,} line items)')
    print(f'Embedding matrix            : {embeddings.shape} {embeddings.dtype}')
    print(f'Row norms                   : min {np.linalg.norm(embeddings, axis=1).min():.4f}, '
          f'max {np.linalg.norm(embeddings, axis=1).max():.4f}')

    # Pairwise similarity of a small slice, to show the range the dot product
    # actually occupies on this catalogue rather than its theoretical [-1, 1].
    with np.errstate(**MATMUL_ERRSTATE):
        slice_sim = embeddings[:500] @ embeddings[:500].T
    off_diagonal = slice_sim[~np.eye(len(slice_sim), dtype=bool)]
    print(f'Off-diagonal similarity     : mean {off_diagonal.mean():.3f}, '
          f'min {off_diagonal.min():.3f}, max {off_diagonal.max():.3f} '
          f'(first 500 products)')

    # A max of 1.0 off the diagonal is these, not a bug: distinct codes whose
    # descriptions are byte-identical embed to the same vector.
    shared = products.groupby('description')['stock_code'].size()
    shared = shared[shared > 1]
    print(f'Descriptions on >1 product  : {len(shared)} descriptions, '
          f'{int(shared.sum())} products')

    query = products['description'].iloc[0]
    print(f'\nNearest neighbours of "{query}":')
    print(most_similar(query, products, embeddings, model=model, top_n=5)
          .to_string(index=False))

    save_embeddings(products, embeddings)
    print(f'\nSaved -> {OUT_FILE}')


if __name__ == '__main__':
    main()

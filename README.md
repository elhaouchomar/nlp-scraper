# NLP Enriched News Intelligence

## Scandal Detection Methodology

**Embeddings Chosen:** 
We use the `all-MiniLM-L6-v2` model from the `sentence-transformers` library. This is a highly efficient pre-trained model for mapping sentences to a 384-dimensional dense vector space. It allows us to capture the *semantic meaning* of an environmental disaster rather than relying strictly on exact keyword matches, which reduces false negatives.

**Distance/Similarity Chosen:** 
We use **Cosine Similarity** to compute the distance between the disaster keyword embeddings and the article sentence embeddings. Cosine similarity measures the angle between vectors rather than their magnitude. This is ideal for text analysis because it accurately determines if a sentence is semantically pointing in the same direction as a "scandal," regardless of how long or short the sentence is.
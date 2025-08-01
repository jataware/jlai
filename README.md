# jlai

Playing w/ scaling out via modal

## Embeddings

Embeddings via [text-embeddings-inference](https://github.com/huggingface/text-embeddings-inference):
```
# --
# dev
JLAI_MAX_CONTAINERS=10 JLAI_GPU_TYPE=A10G modal run -m jlai.embed.tei --test
# or
JLAI_MAX_CONTAINERS=10 JLAI_GPU_TYPE=A10G modal run -m jlai.embed.tei --inpath data.jl --outpath embeddings.npy
# `data.jl` has lines like `{"test" : "i am a sentence"}`

# --
# prod
JLAI_MAX_CONTAINERS=10 JLAI_GPU_TYPE=A10G modal deploy -m jlai.embed.tei
# ... and then can `pip install jlai` and you can call embed_dataset from anywhere and it does GPUs / autoscaling ...
```

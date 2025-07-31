# jlai

Playing w/ scaling out via modal

## Embeddings

Embeddings via [text-embeddings-inference](https://github.com/huggingface/text-embeddings-inference):
```
JLAI_MAX_CONTAINERS=10 JLAI_GPU_TYPE=A10G modal run -m jlai.embed.tei --inpath data.jl --outpath embeddings.jl
```

`data.jl` has lines like
```
{"test" : "i am a sentence"}
...
```

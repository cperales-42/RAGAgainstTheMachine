*This project has been created as part of the 42 curriculum by caperale.*

# RAG against the machine

Retrieval-Augmented Generation (RAG) pipeline that answers questions about the
vLLM codebase: it ingests the repository, indexes it, retrieves the most
relevant snippets for a question, and generates a grounded answer with
`Qwen/Qwen3-0.6B`.

## Instructions

```bash
uv sync
```

See `Makefile` for `install`, `run`, `debug`, `clean`, `lint` and
`lint-strict` rules.

## Resources

- [vLLM documentation](https://docs.vllm.ai/)
- [Pydantic documentation](https://docs.pydantic.dev/)
- [BM25 explained](https://en.wikipedia.org/wiki/Okapi_BM25)
- [Qwen3 model card](https://huggingface.co/Qwen/Qwen3-0.6B)

Full documentation (architecture, chunking strategy, retrieval method,
performance analysis, design decisions, challenges and example usage) will be
completed in a later phase of the project.

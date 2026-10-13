# Dify Agent

Agenton documentation lives in [`docs/agenton/guide/index.md`](docs/agenton/guide/index.md) and
[`docs/agenton/examples/index.md`](docs/agenton/examples/index.md).

Dify Agent runtime documentation lives in [`docs/dify-agent/index.md`](docs/dify-agent/index.md).
Build all docs with `make docs` from this directory.

### Knowledge source handles

Generated-query `knowledge_base_search` results with native dataset, document
and segment identities include a JSON `claimcheck_sources` envelope beside
readable result text. Each retrieved passage gets a UUID handle that survives
the existing tool-observation persistence. Consumers should accept the versioned
envelope only from this tool, validate identities and recheck access before
opening originals. Result positions are not PDF page numbers. External results
without reliable IDs retain the existing readable observation.

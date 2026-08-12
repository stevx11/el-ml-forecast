# data/

On-disk artifacts. Contents are gitignored; the directory structure is tracked.

| Directory | Contents |
|---|---|
| `raw/` | Unmodified provider payloads, kept for replay and debugging |
| `processed/` | Cleaned intermediate tables |
| `features/` | Exported training matrices (parquet) |
| `models/` | Serialised model artifacts, referenced by `model_versions.artifact_path` |

The database is the source of truth for everything the application reads at
runtime. Files here are derived and can be regenerated; nothing in the system
depends on them surviving.

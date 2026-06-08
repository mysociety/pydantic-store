# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

[comment]: # (Template for updates)
## [0.2.0] - 2026-06-07
### Added
- `PydanticDBM.query(predicate, mode=...)` for filtering stored objects with a lambda on the model, returned as a live `ValuesView`. `mode="sql"` (default) type-checks the lambda against the real model but runs it once against an expression-builder stand-in, translating it into a single parameterised SQLite query via `json_extract`/`json_each` — supports comparisons, `&`/`|`/`~` (AND/OR/NOT), `.contains`/`.startswith`/`.endswith`, and the `is_in()` helper for membership tests matching Django's `__in=[...]` convention (`is_in(m.value, [1, 2, 3])`), with nested-path navigation through models and dicts. `mode="filter"` instead runs the lambda as an ordinary Python predicate against each deserialised record — supports arbitrary Python logic (`and`/`or`/`not`, string/collection methods, computed properties, ...) at the cost of fetching and deserialising every record.
### Changed
- `PydanticDBM.values()`, `.items()` and `.query()` now return live `ValuesView`/`ItemsView` objects backed by a single query per iteration, instead of the inherited `MutableMapping` defaults that issue one query per key.

## [0.1.0] - 2025-11-13
### Added
- Initial release.

## [x.x.x] - YYYY-MM-DD
### Added
- Anything added since last version
### Changed
- Anything changed from last version
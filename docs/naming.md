# Names and versions

Rule: **names say what a thing is, tags say which version.** No `-v2`, dates or author names in repository, package or
dataset names.

| Component | Name | Where | Versioning |
|---|---|---|---|
| Project (paper, dataset configs, toolkit, docs) | `WildfireX-SLAM` | GitHub `zhicongsun/WildfireX-SLAM` | dataset releases `v1.0` (paper), `v2.0`, ... as git tags |
| Toolkit | Python package / command `wildfirex` | `src/wildfirex` in this repository, PyPI `wildfirex` | SemVer `0.x` → `1.0` |
| Dataset | WildfireX-SLAM | Hugging Face `zhicongsun/WildfireX-SLAM` (v2.0); v1.0 on request | `v1.0` (paper), `v2.0`, `v3.0` (dynamic fire with fire-front ground truth) |
| Simulator (packaged UE environment) | WildfireX-Sim | Hugging Face `zhicongsun/WildfireX-Sim` (coming soon) | `1.0` = paper; `1.1` = + fire kits, moon (v2.0); `2.0` = dynamic fire with fire-front ground truth |

- Toolkit, simulator and dataset are versioned independently; every `meta.json` records the toolkit, config-schema,
  simulator and dataset versions it was produced with.
- Config files carry `schema_version`; new optional fields (e.g. `fire.spread`) do not bump it.
- Sequence ids are `<site>-<number>` (e.g. `ed04-0002`): `ed` = Electric Dreams map, `01`… = fire site, `0001`… = running
  number. Conditions are metadata fields, so adding conditions never renames data.
- World positions (sites, surface maps) are stored in UE world coordinates; AirSim NED origins differ between builds.
- Use a Zenodo *concept DOI* so all versions of the dataset share one citation.

## Later

- If a second project builds on the toolkit, the toolkit can move to its own repository. The package name `wildfirex`
  stays, so `pip install wildfirex` and `import wildfirex` keep working.
- Moving to a lab organisation: transfer the repositories (GitHub and Hugging Face) **once**, keeping their names.
  GitHub keeps stars, issues and forks and redirects old URLs as long as no new repository is created at the old location.

# Releases

One page per tagged release, newest first. A page states what the release includes, what it does
not include, known issues, verification output, and where the image runs. The first release also
lists every command, route, capability and environment variable.

A release has one `vX.Y.Z` name shared by the Git tag, the GitHub release and the private Docker Hub
image tag `oramasearch/agent-autosuggest-corpus:vX.Y.Z`. The image tag names the digest already
built and verified; it is never produced by a second build.

| Release | Date | Summary |
|---|---|---|
| [v0.1.0](v0.1.0.md) | 2026-10-17 | The first release: the corpus generator CLI and replay, the Lake library agent (`/healthz`, `/v1/turn`, the agent bus), prompt v3, the contamination guard, the methodology and solution registries, and the private image |

`pyproject.toml`'s `[project].version` is the one place the version is set; `GET /healthz` reports
it, and `tests/test_release.py` fails when that version has no page here.

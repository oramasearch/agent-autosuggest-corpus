# syntax=docker/dockerfile:1
# SPDX-License-Identifier: MIT

# The Auto-Suggest Corpus Generator as a lake library agent (lake#263): private on Docker Hub as
# oramasearch/agent-autosuggest-corpus. The eval set is a named build context, `evals`
# (agent-eval-generator/evals/profiles): without it the contamination guard cannot run, and the
# service answers nothing.
FROM python:3.12-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable

FROM python:3.12-slim
RUN useradd --uid 10001 --create-home agent
WORKDIR /app
COPY --from=build /app/.venv /app/.venv
COPY --from=evals . /app/evals/profiles/
ENV PATH=/app/.venv/bin:$PATH \
    EVAL_PROFILES=/app/evals/profiles \
    PORT=8080
USER 10001
EXPOSE 8080
ENTRYPOINT ["agent-autosuggest-corpus-serve"]

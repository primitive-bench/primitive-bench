"""Typed configuration loaded from the environment (.env)."""
from __future__ import annotations

from functools import lru_cache

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # registry pump
    sec_edgar_user_agent: str = Field(default="arlen-bench arlen1788@berkeley.edu")
    nvd_api_key: str = Field(default="")
    github_token: str = Field(default="")
    github_release_repos: str = Field(default="")
    fr_api_base: str = Field(default="https://www.federalregister.gov/api/v1")

    # search vendors (web_search primitive). Each key is read from its bare
    # <VENDOR>_API_KEY environment variable.
    exa_api_key: str = ""
    brave_search_api_key: str = ""
    tavily_api_key: str = ""
    serpapi_key: str = ""
    perplexity_api_key: str = ""
    google_cse_key: str = ""
    google_cse_engine_id: str = ""
    bing_search_key: str = ""
    you_api_key: str = ""

    # extraction vendors (web_extraction primitive).
    firecrawl_api_key: str = ""
    jina_api_key: str = ""
    apify_api_key: str = ""
    apify_actor: str = "apify/website-content-crawler"
    brightdata_api_key: str = ""

    # split secret — never serialized into any artifact
    hmac_salt: str = Field(default="")

    # sentinel
    sentinel_base_url: str = "https://sentinel.arlenkumar.com"
    sentinel_publish_branch: str = "gh-pages"

    # knobs
    split_public_fraction: float = 0.70
    probe_repetitions: int = 3
    liveness_window_hours: int = 6
    verify_min_gap_hours: int = 6
    http_timeout_seconds: float = 30.0
    duckdb_path: str = "data/eval.duckdb"

    def repos(self) -> list[str]:
        return [r.strip() for r in self.github_release_repos.split(",") if r.strip()]

    def require_salt(self) -> str:
        if not self.hmac_salt:
            raise RuntimeError("HMAC_SALT is unset. Generate with `openssl rand -hex 32`.")
        return self.hmac_salt


@lru_cache
def get_settings() -> Settings:
    return Settings()

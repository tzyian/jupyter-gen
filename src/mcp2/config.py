from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    tavily_api_key: str | None = Field(default=None, alias="TAVILY_API_KEY")
    elevenlabs_api_key: str | None = Field(default=None, alias="ELEVENLABS_API_KEY")

    sentence_model: str = Field(default="all-MiniLM-L6-v2", alias="SENTENCE_MODEL")
    index_dir: str = Field(default="data/index", alias="INDEX_DIR")

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()

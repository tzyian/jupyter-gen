from pydantic_settings import BaseSettings
from pydantic import Field, SecretStr


class Settings(BaseSettings):
    openai_api_key: SecretStr | None = Field(default=None, alias="OPENAI_API_KEY")
    tavily_api_key: SecretStr | None = Field(default=None, alias="TAVILY_API_KEY")
    elevenlabs_api_key: SecretStr | None = Field(
        default=None, alias="ELEVENLABS_API_KEY"
    )
    langfuse_public_key: SecretStr | None = Field(
        default=None, alias="LANGFUSE_PUBLIC_KEY"
    )
    langfuse_secret_key: SecretStr | None = Field(
        default=None, alias="LANGFUSE_SECRET_KEY"
    )
    langfuse_base_url: str | None = Field(default=None, alias="LANGFUSE_BASE_URL")

    sentence_model: str = Field(default="all-MiniLM-L6-v2", alias="SENTENCE_MODEL")
    index_dir: str = Field(default="data/index", alias="INDEX_DIR")

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()

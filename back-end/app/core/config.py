from pydantic import BaseSettings

class Settings(BaseSettings):
    DB_USER: str = "fastteamjpsm"
    DB_PASSWORD: str = "fast!jpsm"
    DB_HOST: str = "localhost"
    DB_PORT: int = 3306
    DB_NAME: str = "fast_db"

    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    @property
    def DATABASE_URL(self) -> str:
        return (
            f"mysql+pymysql://{self.DB_USER}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )

    class Config:
        env_file = ".env"

settings = Settings()
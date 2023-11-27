import asyncio

import pytest
import pytest_asyncio
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from qqqr.utils.net import ClientAdapter


class test_env(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="test_")
    uin: int = 0
    password: SecretStr = Field(default="")


@pytest.fixture(scope="session")
def env():
    return test_env()


@pytest.fixture(scope="module")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.run_until_complete(loop.shutdown_asyncgens())
    loop.close()


@pytest_asyncio.fixture(scope="module")
async def client():
    async with ClientAdapter() as client:
        yield client

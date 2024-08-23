import pytest
from httpx import AsyncClient
from server import Input, app


@pytest.mark.parametrize(
    "input",
    [
        Input(
            llm="openai",
            retriever="textchunk",
            rag="base",
            character="해리",
            prompt="Hello",
        )
    ],
)
@pytest.mark.asyncio
async def test_invoke(input: Input) -> None:
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post("/invoke", json=input.model_dump())

    assert response.status_code == 200
    assert "generation" in response.json()

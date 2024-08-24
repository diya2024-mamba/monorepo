import pytest
from httpx import AsyncClient
from server import InvokeInput, RandomInput, VoteConfig, VoteInput, app


@pytest.mark.parametrize(
    "input",
    [
        InvokeInput(
            llm="openai",
            retriever="textchunk",
            rag="base",
            character="해리",
            prompt="Hello",
        )
    ],
)
@pytest.mark.asyncio
async def test_invoke(input: InvokeInput) -> None:
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post("/invoke", json=input.model_dump())

    assert response.status_code == 200
    assert "generation" in response.json()


@pytest.mark.parametrize(
    "input",
    [
        RandomInput(
            character="해리",
            prompt="Hello",
        )
    ],
)
@pytest.mark.asyncio
async def test_random(input: RandomInput) -> None:
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post("/random", json=input.model_dump())

    assert response.status_code == 200

    output = response.json()
    assert "A" in output
    assert "llm" in output["A"]
    assert "generation" in output["A"]["output"]


@pytest.mark.parametrize(
    "input",
    [
        VoteInput(
            winner=VoteConfig(
                llm="openai",
                retriever="textchunk",
                rag="base",
            ),
            loser=VoteConfig(
                llm="openai",
                retriever="metadata",
                rag="base",
            ),
        )
    ],
)
@pytest.mark.asyncio
async def test_vote(input: VoteInput) -> None:
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post("/vote", json=input.model_dump())

    assert response.status_code == 200
    assert "message" in response.json()

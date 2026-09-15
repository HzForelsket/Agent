# Copyright (c) Microsoft. All rights reserved.

"""Outer rollout adapters for the unchanged Spider and CrewAI 20 Questions workflows."""

import asyncio
import sys
from pathlib import Path
from typing import Any, Callable

import agentlightning as agl

EXAMPLES = Path(__file__).resolve().parent.parent


def make_agent(config: dict[str, Any], record: Callable[[str, dict[str, Any]], None]) -> agl.LitAgent[Any]:
    """Route model endpoints and persist outcomes without modifying prompts or graph/flow nodes."""
    if config["agent"] == "sql":
        sys.path.insert(0, str(EXAMPLES / "spider"))
        from sql_agent import LitSQLAgent

        class RecordedSQLAgent(LitSQLAgent):
            async def rollout_async(self, task: Any, resources: Any, rollout: Any) -> float:
                # The original synchronous LangGraph workflow must not block the HTTP recording proxy.
                self.spider_dir = task["spider_dir"]
                reward = await asyncio.to_thread(super().rollout, task, resources, rollout)
                if reward is None:
                    raise RuntimeError("Original SQL workflow returned no result; inspect worker log")
                record("events.jsonl", {"event": "workflow_completed", "agent": "sql", "reward": reward})
                return reward

        return RecordedSQLAgent()

    sys.path.insert(0, str(EXAMPLES / "tinker"))
    from crewai import LLM as CrewLLM
    from q20_agent import AnswererResponse, SearchTool, TwentyQuestionsFlow

    class RecordedQ20Agent(agl.LitAgent[Any]):
        async def rollout_async(self, task: Any, resources: Any, rollout: Any) -> float:
            llm = resources["main_llm"]
            endpoint = llm.endpoint.removesuffix("/v1")
            player = CrewLLM(model="openai/" + llm.model, base_url=llm.endpoint, api_key="dummy", timeout=120.0)
            answerer = CrewLLM(
                model="openai/" + config["model"],
                base_url=endpoint + "/answerer/v1",
                api_key="dummy",
                reasoning_effort="low",
                response_format=AnswererResponse,
                timeout=120.0,
            )
            search = (
                SearchTool(
                    model=CrewLLM(
                        model="openai/" + config["model"],
                        base_url=endpoint + "/search/v1",
                        api_key="dummy",
                        reasoning_effort="none",
                        timeout=120.0,
                    )
                )
                if config["q20_search"]
                else None
            )
            flow = TwentyQuestionsFlow(player_llm=player, answer_llm=answerer, search_tool=search)
            # Execute the original Flow on a thread: CrewAI nodes contain synchronous HTTP calls.
            await asyncio.to_thread(
                lambda: asyncio.run(flow.kickoff_async({"answer": task["answer"], "category": task["category"]}))
            )
            record("events.jsonl", {"event": "workflow_completed", "agent": "q20", "state": flow.state.model_dump()})
            return 1.0 if flow.state.correct else 0.0

    return RecordedQ20Agent()


if __name__ == "__main__":
    import argparse
    import json
    from importlib.metadata import version

    parser = argparse.ArgumentParser(description="Check original workflow imports before starting NPU serving.")
    parser.add_argument("--agent", choices=("sql", "q20"), required=True)
    args = parser.parse_args()
    agent = make_agent(vars(args), lambda name, value: print(json.dumps(value)))
    packages = ("langchain", "langgraph", "langchain-openai") if args.agent == "sql" else ("crewai",)
    print(
        json.dumps(
            {"agent": args.agent, "adapter": type(agent).__name__, "packages": {p: version(p) for p in packages}}
        )
    )

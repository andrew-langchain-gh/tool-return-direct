"""Reproduce the A2UI `return_direct` finding.

Runs two agents against the live API and counts model calls per turn:

  working  return_direct tool + after_agent middleware  -> 1 call
  broken   AIMessage returned inside the tool's Command -> 2 calls (400 on Anthropic)

Requires ANTHROPIC_API_KEY in .env. See langgraph-a2ui-return-direct-handoff.md.
"""

import operator
from typing import Annotated, Any, NotRequired

from dotenv import load_dotenv
from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import after_agent
from langchain.messages import AIMessage, ToolMessage
from langchain.tools import ToolRuntime, tool
from langchain_core.callbacks import BaseCallbackHandler
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_stream_writer
from langgraph.runtime import Runtime
from langgraph.types import Command

load_dotenv(override=True)

MODEL = "anthropic:claude-sonnet-4-6"
QUESTION = "Show me the summary for account 998877."
A2UI_DIRECT_TOOLS = {"fetch_account_summary"}


class A2UIState(AgentState):
    a2ui_surfaces: NotRequired[Annotated[list[dict[str, Any]], operator.add]]


def load_account_summary(account_id: str) -> dict[str, Any]:
    return {
        "account_id": account_id,
        "nickname": "Everyday Checking",
        "available_balance": 4820.17,
        "pending_charges": 132.40,
    }


def build_a2ui_surface(data: dict[str, Any]) -> dict[str, Any]:
    return {
        "surfaceId": f"account-summary-{data['account_id']}",
        "component": "AccountSummaryCard",
        "props": data,
    }


def render_summary(surface: dict[str, Any]) -> str:
    """Deterministic restatement of the surface data. No model involved."""
    props = surface["props"]
    return (
        f"Displayed the account summary for {props['account_id']} "
        f"({props['nickname']}): available balance ${props['available_balance']:,.2f}, "
        f"pending charges ${props['pending_charges']:,.2f}."
    )


@tool(return_direct=True)
def fetch_account_summary(account_id: str, runtime: ToolRuntime) -> Command:
    """Fetch an account summary and render it as an A2UI surface."""
    surface = build_a2ui_surface(load_account_summary(account_id))

    # Push to the UI immediately — the existing queue flush goes here.
    get_stream_writer()({"type": "a2ui_surface", "surface": surface})

    return Command(
        update={
            "a2ui_surfaces": [surface],
            "messages": [
                ToolMessage(
                    content=render_summary(surface),
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


@after_agent(state_schema=A2UIState)
def append_a2ui_summary(state: A2UIState, runtime: Runtime) -> dict[str, Any] | None:
    """Restate a return-direct A2UI result as an AIMessage, with no model call."""
    last = state["messages"][-1]
    if not isinstance(last, ToolMessage) or last.name not in A2UI_DIRECT_TOOLS:
        return None
    return {"messages": [AIMessage(content=render_summary(state["a2ui_surfaces"][-1]))]}


@tool(return_direct=True)
def fetch_account_summary_broken(account_id: str, runtime: ToolRuntime) -> Command:
    """Fetch an account summary and render it as an A2UI surface."""
    surface = build_a2ui_surface(load_account_summary(account_id))
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=render_summary(surface),
                    tool_call_id=runtime.tool_call_id,
                ),
                # The message that cancels return_direct. See the handoff doc.
                AIMessage(content=render_summary(surface)),
            ]
        }
    )


class ModelCallCounter(BaseCallbackHandler):
    def __init__(self) -> None:
        self.count = 0

    def on_chat_model_start(self, serialized, messages, **kwargs) -> None:
        self.count += 1


def check_working_route() -> None:
    print("WORKING ROUTE — after_agent middleware")

    agent = create_agent(
        model=MODEL,
        tools=[fetch_account_summary],
        middleware=[append_a2ui_summary],
        state_schema=A2UIState,
        checkpointer=InMemorySaver(),
    )

    counter = ModelCallCounter()
    config = {"configurable": {"thread_id": "working"}, "callbacks": [counter]}

    surfaces = list(
        agent.stream(
            {"messages": [{"role": "user", "content": QUESTION}]},
            config=config,
            stream_mode="custom",
        )
    )

    messages = agent.get_state(config).values["messages"]
    last = messages[-1]

    print(f"  model calls in turn 1:    {counter.count}")
    print(f"  surfaces pushed mid-tool: {len(surfaces)}")
    print(f"  thread history:           {[type(m).__name__ for m in messages]}")
    print(f"  final message:            {last.content!r}")

    assert counter.count == 1, f"expected 1 model call, got {counter.count}"
    assert len(surfaces) == 1
    assert isinstance(last, AIMessage) and not last.tool_calls
    assert last.content == render_summary(build_a2ui_surface(load_account_summary("998877")))

    followup = agent.invoke(
        {"messages": [{"role": "user", "content": "What was the pending charges figure?"}]},
        config={"configurable": {"thread_id": "working"}},
    )
    answer = str(followup["messages"][-1].content)
    print(f"  turn 2 recall:            {answer!r}")
    assert "132" in answer, "turn 2 could not recall the turn-1 data"


def check_broken_route() -> None:
    print("\nBROKEN ROUTE — AIMessage inside the tool's Command")

    agent = create_agent(
        model=MODEL,
        tools=[fetch_account_summary_broken],
        checkpointer=InMemorySaver(),
    )

    counter = ModelCallCounter()
    try:
        agent.invoke(
            {"messages": [{"role": "user", "content": QUESTION.replace("998877", "112233")}]},
            config={"configurable": {"thread_id": "broken"}, "callbacks": [counter]},
        )
        outcome = "completed (provider tolerated assistant-final history)"
    except Exception as exc:  # noqa: BLE001
        outcome = f"{type(exc).__name__}: {str(exc)[:120]}"

    print(f"  model calls in turn 1:    {counter.count}")
    print(f"  outcome:                  {outcome}")

    assert counter.count == 2, f"expected the model to be re-entered, got {counter.count} call(s)"


def main() -> None:
    check_working_route()
    check_broken_route()
    print("\nAll assertions passed.")


if __name__ == "__main__":
    main()

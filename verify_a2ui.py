"""Reproduce the A2UI `return_direct` finding.

Runs three agents against the live API and counts model calls per turn:

  working  return_direct tool + after_agent middleware  -> 1 call
  broken   AIMessage returned inside the tool's Command -> 2 calls (400 on Anthropic)
  failure  A2UI tool errors after an earlier success    -> no stale surface restated

Requires ANTHROPIC_API_KEY in .env. See langgraph-a2ui-return-direct-handoff.md.
"""

import operator
from collections.abc import Callable, Sequence
from typing import Annotated, Any, NotRequired

from dotenv import load_dotenv
from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import AgentMiddleware, after_agent, wrap_tool_call
from langchain.messages import AIMessage, AnyMessage, ToolMessage
from langchain.tools import ToolRuntime, tool
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.runnables import RunnableConfig
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
    surface["toolCallId"] = runtime.tool_call_id

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


def tool_messages_from_last_turn(messages: Sequence[AnyMessage]) -> list[ToolMessage]:
    """The ToolMessages appended after the most recent AIMessage."""
    collected: list[ToolMessage] = []
    for message in reversed(messages):
        if not isinstance(message, ToolMessage):
            break
        collected.append(message)
    return collected


@after_agent(state_schema=A2UIState)
def append_a2ui_summary(state: A2UIState, runtime: Runtime) -> dict[str, Any] | None:
    """Restate return-direct A2UI results as an AIMessage, with no model call."""
    rendered_tool_call_ids = {
        message.tool_call_id
        for message in tool_messages_from_last_turn(state["messages"])
        if message.name in A2UI_DIRECT_TOOLS and message.status != "error"
    }
    if not rendered_tool_call_ids:
        return None

    surfaces = [
        surface
        for surface in state.get("a2ui_surfaces", [])
        if surface["toolCallId"] in rendered_tool_call_ids
    ]
    if not surfaces:
        return None

    return {"messages": [AIMessage(content=" ".join(render_summary(s) for s in surfaces))]}


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


@tool("fetch_account_summary", return_direct=True)
def fetch_account_summary_unavailable(account_id: str) -> Command:
    """Fetch an account summary and render it as an A2UI surface."""
    raise RuntimeError("account service timed out")


@wrap_tool_call
def convert_tool_errors_to_messages(
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
) -> ToolMessage | Command[Any]:
    """Turn a tool exception into an error ToolMessage, named as ToolNode's own handler names it."""
    try:
        return handler(request)
    except Exception as exc:  # noqa: BLE001
        return ToolMessage(
            content=f"The account service is unavailable ({exc}).",
            tool_call_id=request.tool_call["id"],
            name=request.tool_call["name"],
            status="error",
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
    config: RunnableConfig = {
        "configurable": {"thread_id": "working"},
        "callbacks": [counter],
    }

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


def check_failure_route() -> None:
    print("\nFAILURE ROUTE — A2UI tool errors on a thread that already rendered a surface")

    checkpointer = InMemorySaver()
    config: RunnableConfig = {"configurable": {"thread_id": "failure"}}

    healthy = create_agent(
        model=MODEL,
        tools=[fetch_account_summary],
        middleware=[append_a2ui_summary],
        state_schema=A2UIState,
        checkpointer=checkpointer,
    )
    turn_one = healthy.invoke({"messages": [{"role": "user", "content": QUESTION}]}, config=config)
    stale_surface = turn_one["a2ui_surfaces"][-1]
    stale_balance = f"{stale_surface['props']['available_balance']:,.2f}"
    assert isinstance(turn_one["messages"][-1], AIMessage), "turn 1 did not render a surface"

    # `wrap_tool_call` takes no state_schema, so its middleware is typed against the
    # base AgentState. AgentMiddleware's StateT is invariant — annotate to mix the two.
    degraded_middleware: list[AgentMiddleware[Any, Any, Any]] = [
        convert_tool_errors_to_messages,
        append_a2ui_summary,
    ]

    # Same thread, same tool name — the account service is now down.
    degraded = create_agent(
        model=MODEL,
        tools=[fetch_account_summary_unavailable],
        middleware=degraded_middleware,
        state_schema=A2UIState,
        checkpointer=checkpointer,
    )
    result = degraded.invoke(
        {"messages": [{"role": "user", "content": "Now show me account 112233."}]},
        config=config,
    )

    last = result["messages"][-1]
    account = stale_surface["props"]["account_id"]
    print(f"  turn 1 surface:           {account}, available balance ${stale_balance}")
    print(f"  turn 2 last message:      {type(last).__name__}")
    print(f"  final message:            {str(last.content)!r}")

    assert isinstance(last, ToolMessage) and last.status == "error"
    assert stale_balance not in str(last.content), (
        f"the stale {account} surface leaked into a failed turn"
    )


def main() -> None:
    check_working_route()
    check_broken_route()
    check_failure_route()
    print("\nAll assertions passed.")


if __name__ == "__main__":
    main()

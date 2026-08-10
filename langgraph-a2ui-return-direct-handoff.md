# Handoff: Skipping the ReAct Reasoning Step for A2UI Tool Calls

> Verified end-to-end against `langchain==1.3.14`, `langchain-core==1.5.3`, `langgraph==1.2.10`,
> `langchain-anthropic==1.5.4` (model `anthropic:claude-sonnet-4-6`) on 2026-08-10.

## Context / Scenario

Customer's LangGraph agent flow:

1. Graph calls a native tool.
2. Tool fetches data, builds an **A2UI surface**, and returns it via a `ToolMessage`. The A2UI surface is also pushed to the UI immediately through a Python queue.
3. The queue flushes to the UI — the UI renders the A2UI surface with data right away.
4. The graph loops back into the ReAct loop: the LLM is called again to "review" the tool data. This call is slow.
5. The UI is already showing the surface and is just waiting on the graph to finish — but the LLM's output ends up being basically the same data already shown.

## The Core Question

Can we **skip that last reasoning/LLM call for specific tools** — to avoid the slow, redundant step — **without losing the ability to reference that tool's data in later conversation turns**?

- The `ToolMessage` *is* in thread history already, but apparently isn't sufficient on its own for the LLM to use/search that data in later turns.
- If we skip the LLM review, we lose the `AIMessage` that (today) makes the data usable downstream.

## Answer at a Glance

| Requirement | Mechanism |
|---|---|
| Skip the redundant LLM "review" call | `return_direct=True` on the tool |
| Push the A2UI surface to the UI mid-tool | `get_stream_writer()` from inside the tool (equivalent to the existing queue push) |
| Keep the surface in graph state | Tool returns `Command(update={...})` — **`ToolMessage` only** in `messages` |
| Get an `AIMessage` into thread history, with no model call | An **`after_agent` middleware hook** appends a deterministic `AIMessage` |

The one non-obvious part is the last row. The natural-looking approach — building the `AIMessage` inside the tool and returning it in the same `Command` — **silently defeats `return_direct`**. See [The Pitfall](#the-pitfall-do-not-append-the-aimessage-inside-the-tool) below.

## 1. How `return_direct` Actually Routes

`return_direct=True` short-circuits the agent loop: the tool executes normally, its output is wrapped in a `ToolMessage`, and the agent returns without another model call.

What matters for our design is *how* the graph decides to short-circuit. In `langchain/agents/factory.py`, the conditional edge out of the `tools` node (`_make_tools_to_model_edge`) does this:

```python
last_ai_message, tool_messages = _fetch_last_ai_and_tool_messages(state["messages"])
if last_ai_message is None:
    return model_destination

client_side_tool_calls = [
    c for c in last_ai_message.tool_calls if c["name"] in tool_node.tools_by_name
]
if client_side_tool_calls and all(
    tool_node.tools_by_name[c["name"]].return_direct for c in client_side_tool_calls
):
    return end_destination          # <- the short-circuit
...
return model_destination            # <- the redundant LLM call
```

And `_fetch_last_ai_and_tool_messages` scans **backwards through `state["messages"]` for the last `AIMessage`**.

Two consequences drive everything below:

- The decision is made on the **message list as it stands after the tools node runs**. Whatever the tool wrote into state is already visible.
- The decision reads only the last `AIMessage`'s `tool_calls` and the tool registry. It never inspects the `Command` the tool returned, nor the `ToolMessage` content.

Also note: **`return_direct` takes effect only when *all* tools called in that turn have `return_direct=True`.** One ordinary tool in a parallel batch sends the whole turn back through the model.

## The Pitfall: Do Not Append the `AIMessage` Inside the Tool

This is the approach that looks right and does not work:

```python
# DO NOT DO THIS
@tool(return_direct=True)
def fetch_a2ui_data(query: str, runtime: ToolRuntime) -> Command:
    ...
    return Command(update={"messages": [
        ToolMessage(content=data, tool_call_id=runtime.tool_call_id),
        AIMessage(content=summarize(data)),   # <- breaks return_direct
    ]})
```

Walk it through the routing logic above. After the tools node, `state["messages"]` ends:

```
..., AIMessage(tool_calls=[fetch_a2ui_data]), ToolMessage(...), AIMessage(synthetic)
```

`_fetch_last_ai_and_tool_messages` now returns the **synthetic** `AIMessage` as `last_ai_message`. It has no `tool_calls`, so `client_side_tool_calls` is empty, the `if` is skipped, and the edge falls through to `return model_destination`.

**Net effect: the LLM call you were trying to eliminate still happens.** Worse, it is now invoked on a history ending in an assistant message. Observed behavior in the verification run:

```
[4] broken variant outcome: BadRequestError: Error code: 400 -
    'This model does not support assistant message prefill.
     The conversation must end with a user message.'
[4] model calls with AIMessage inside the tool Command: 2
```

With Anthropic this is a hard 400. With providers that tolerate assistant-final input, it degrades quietly into an extra latency hit plus a continuation of your own synthetic text — the failure mode is invisible until you count model calls.

The fix is not to reorder the two messages. Putting the synthetic `AIMessage` *before* the `ToolMessage` fails the same routing check *and* produces an invalid tool-call/tool-result sequence.

## 2. The Working Implementation

Keep `return_direct=True` on the tool, and inject the synthetic `AIMessage` from an **`after_agent` middleware hook** instead.

Why this is the right seam: `after_agent` middleware becomes the graph's `exit_node`, and the `tools → exit_node` edge is wired precisely *because* a tool has `return_direct=True`. So the hook runs on the return-direct exit path, **after** the routing decision has already been made against unmodified messages. Its state update is merged and persisted by the checkpointer like any other node output.

The tool can still return a `Command` — that is safe, as long as `update["messages"]` contains only the `ToolMessage`.

### State

Extend `AgentState` with the surfaces you want to keep. Give the field a reducer: the model can call tools in parallel, and concurrent writes to the same field need a merge rule.

```python
import operator
from typing import Annotated, Any, NotRequired

from langchain.agents import AgentState


class A2UIState(AgentState):
    a2ui_surfaces: NotRequired[Annotated[list[dict[str, Any]], operator.add]]
```

### The tool

```python
from langchain.messages import ToolMessage
from langchain.tools import ToolRuntime, tool
from langgraph.config import get_stream_writer
from langgraph.types import Command

A2UI_DIRECT_TOOLS = {"fetch_account_summary"}


@tool(return_direct=True)
def fetch_account_summary(account_id: str, runtime: ToolRuntime) -> Command:
    """Fetch an account summary and render it as an A2UI surface."""
    data = load_account_summary(account_id)
    surface = build_a2ui_surface(data)

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
```

`render_summary` is deterministic, template-driven text built from the data the tool already has — no model involved:

```python
def render_summary(surface: dict[str, Any]) -> str:
    props = surface["props"]
    return (
        f"Displayed the account summary for {props['account_id']} "
        f"({props['nickname']}): available balance ${props['available_balance']:,.2f}, "
        f"pending charges ${props['pending_charges']:,.2f}."
    )
```

### The middleware

```python
from langchain.agents.middleware import after_agent
from langchain.messages import AIMessage
from langgraph.runtime import Runtime


@after_agent(state_schema=A2UIState)
def append_a2ui_summary(state: A2UIState, runtime: Runtime) -> dict[str, Any] | None:
    """Restate a return-direct A2UI tool result as an AIMessage, with no model call."""
    last = state["messages"][-1]
    if not isinstance(last, ToolMessage) or last.name not in A2UI_DIRECT_TOOLS:
        return None
    return {"messages": [AIMessage(content=render_summary(state["a2ui_surfaces"][-1]))]}
```

The guard matters: `after_agent` runs at the end of **every** turn, including ordinary turns that already ended with a real model-generated `AIMessage`. Returning `None` there leaves those untouched.

### Wiring

```python
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
    model="anthropic:claude-sonnet-4-6",
    tools=[fetch_account_summary],
    middleware=[append_a2ui_summary],
    state_schema=A2UIState,
    checkpointer=InMemorySaver(),
)
```

### Verified behavior

Turn 1 — `"Show me the summary for account 998877."`

```
model calls in turn 1:       1          # the tool-calling call only; no review call
surfaces pushed mid-tool:    1
final message type:          AIMessage
final message content:       'Displayed the account summary for 998877 (Everyday Checking):
                              available balance $4,820.17, pending charges $132.40.'
thread history:              [HumanMessage, AIMessage, ToolMessage, AIMessage]
```

Turn 2 on the same `thread_id` — `"What was the pending charges figure?"`

```
'Based on the account summary retrieved, the pending charges for account 998877
 (Everyday Checking) were **$132.40**.'
```

One model call for the turn that renders the surface, and the data is still fully referenceable afterwards.

## 3. Caveats

- **All-or-nothing per turn.** `return_direct` short-circuits only if *every* tool called in that turn has `return_direct=True`. Mixing an A2UI tool with an ordinary tool in one model turn reverts to the normal loop.
- **The synthetic `AIMessage` must have no `tool_calls`**, and must come after the `ToolMessage`. That ordering (`AIMessage(tool_calls)` → `ToolMessage` → `AIMessage`) is valid input for both Anthropic and OpenAI on the next turn.
- **`after_agent` fires on every turn.** Guard on the last message, as above, or you will append synthetic text to normal conversational turns.
- **`get_stream_writer()` requires a LangGraph execution context.** A tool that calls it cannot be invoked standalone in a unit test — inject the writer or guard the call if you need direct-invocation tests.
- **Count model calls, don't eyeball latency.** The pitfall above is only reliably visible by instrumenting `on_chat_model_start`. A `BaseCallbackHandler` counter passed via `config={"callbacks": [...]}` is enough:

  ```python
  class ModelCallCounter(BaseCallbackHandler):
      def __init__(self) -> None:
          self.count = 0

      def on_chat_model_start(self, serialized, messages, **kwargs) -> None:
          self.count += 1
  ```

## 4. Open Question to Resolve Next

Confirm exactly how "search in later turns of the conversation" is implemented on the customer's side:

- If it's just full thread/message history passed back into the LLM, the synthetic `AIMessage` above is sufficient — and the verification run confirms the model recalls the data from it.
- If it's a separate retrieval/embedding search system, it may be worth having it index `ToolMessage` content directly. That removes the need to fabricate an `AIMessage` at all, and the tool + `return_direct=True` alone would do the job — no middleware.

Worth asking either way, since the second answer makes this simpler rather than harder.

## Reference Links

1. Return directly from a tool (Python): https://docs.langchain.com/oss/python/langchain/tools#return-directly-from-a-tool
2. Return directly from a tool (JS): https://docs.langchain.com/oss/javascript/langchain/tools#return-directly-from-a-tool
3. Update state / `Command` from a tool (Python): https://docs.langchain.com/oss/python/langchain/tools#update-state
4. AI message (Python): https://docs.langchain.com/oss/python/langchain/messages#ai-message
5. Middleware hooks, incl. `after_agent` (Python): https://docs.langchain.com/oss/python/langchain/middleware
6. Streaming custom updates from tools: https://docs.langchain.com/oss/python/langchain/streaming#custom-updates
7. Routing source of truth — `_make_tools_to_model_edge`: https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/agents/factory.py

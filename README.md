# tool-return-direct

Investigation into skipping the redundant ReAct "review" LLM call for tools that
return a rendered A2UI surface, without losing the ability to reference that
tool's data in later conversation turns.

## The question

The agent calls a tool, the tool builds an A2UI surface and pushes it to the UI
immediately, and the surface renders. The graph then loops back into the ReAct
loop for one more model call to "review" the tool output — slow, and redundant,
since the UI is already showing the data.

Can that call be skipped for specific tools, while keeping an `AIMessage` in
thread history that later turns can reason over?

## The finding

Yes — but the obvious approach doesn't work, and fails quietly.

Setting `return_direct=True` and returning a `Command` whose `messages` update
contains both the `ToolMessage` and a hand-built `AIMessage` **cancels the
short-circuit**. The conditional edge out of the tools node
(`_make_tools_to_model_edge`) scans backwards for the last `AIMessage` and
short-circuits only if that message's `tool_calls` are all return-direct tools.
The synthetic `AIMessage` becomes the last one, has no `tool_calls`, and the
edge routes back to the model.

The model call you were eliminating still happens — now on a history ending in
an assistant turn. On Anthropic that is a hard 400; on providers that tolerate
assistant-final input it degrades silently into extra latency.

## The fix

Keep `return_direct=True` on the tool, with **only** the `ToolMessage` in its
`Command` update. Append the synthetic `AIMessage` from an `after_agent`
middleware hook instead — that hook is the graph's exit node, so it runs after
the routing decision was made against unmodified messages.

Measured against the live API: **1 model call instead of 2**, with turn-2 recall
of the turn-1 data intact.

## Contents

| File | What it is |
| --- | --- |
| `langgraph-a2ui-return-direct-handoff.md` | Full engineering handoff — routing mechanism, the pitfall, complete working code, caveats |
| `a2ui-return-direct-summary.html` | Readable customer-facing summary of the same material |

## Verified against

`langchain==1.3.14` · `langchain-core==1.5.3` · `langgraph==1.2.10` ·
`langchain-anthropic==1.5.4` · model `anthropic:claude-sonnet-4-6` (10 Aug 2026)

Findings were confirmed by running both routes against the live API with a
`BaseCallbackHandler` counting `on_chat_model_start`. Latency alone will not
surface this — the call count has to be instrumented.

## Setup

```bash
uv sync
cp .env.example .env   # then fill in ANTHROPIC_API_KEY and LANGSMITH_API_KEY
```

This repo holds the investigation and its write-ups, not a deployable agent. The
code in the handoff doc is reference material to lift into the customer's own
graph.

## Open question

How is "search in later turns" implemented on the customer's side? It changes
the recommendation:

- **Full thread history replayed to the model** — the synthetic `AIMessage` is
  the right fix, as documented.
- **A separate retrieval or embedding index** — point it at `ToolMessage`
  content directly. Then no fabricated message and no middleware is needed:
  the tool plus `return_direct=True` is the whole solution.

Worth resolving early, since the second answer removes code rather than adding
it.

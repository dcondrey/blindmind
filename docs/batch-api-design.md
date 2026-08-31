# Design note: opt-in Batch API mode (not implemented)

Status: design only, no code. This documents where a batch-mode provider would
plug into `blindmind/llm.py`'s existing provider abstraction, for headless
scripts specifically (e.g. `headless_evolve_rule30.py`). It is not built because
it's a real design decision (queueing semantics, partial-failure handling, cost
tradeoffs) that should be signed off on deliberately rather than guessed at.

## Why it doesn't belong in the interactive path

Anthropic's Message Batches API (and equivalents on other providers) trades
latency for cost: jobs are typically minutes to up to 24h turnaround, not
seconds. `LLMEngine._completion` is called from `EvolutionEngine._create_candidate`
and awaited directly inside a live `run_generation_cycle` loop that the
interactive `e` (Evolve) menu command drives with a spinner in front of the
user. There's no way to make that read as batch mode without either blocking the
menu for hours or turning "evolve" into a fire-and-forget job the user has to
come back and poll manually -- a materially different UX than what the menu
promises today. Headless scripts (`headless_evolve_rule30.py` and friends) are
the opposite case: nobody is watching a spinner, and they already run
unattended for however long a generation takes. That's the only fit.

## Where it would plug in

`LLMEngine` (`blindmind/llm.py`) already has a clean provider-pool abstraction:
`_get_available_providers()` builds a list of `{model, api_key, label}` dicts,
and `_completion()`/`_try_provider()` iterate that pool with fallback. A batch
provider does not fit that per-call abstraction as-is, because a single
Anthropic Batches API job is not "one call" -- it's a submit step (given N
prompts) and a separate poll/retrieve step, with results keyed by a
per-request `custom_id`, not returned inline. The natural seam is one level up,
at `EvolutionEngine.run_generation_cycle` (`blindmind/engine.py`):

- **Today**: `run_generation_cycle` calls `asyncio.gather(*[self._create_candidate(generation) for _ in range(batch_size)])`, where each `_create_candidate` does two sequential *awaited* LLM round-trips (`generate_mutation` then, if it survives the local pre-filter, `critique_mutation`), each going through `LLMEngine._completion` and returning synchronously.
- **Batch mode**: a `BatchLLMEngine` (or a `batch=True` flag threaded through `LLMEngine`) would need to split candidate creation into two phases instead of one interleaved per-candidate loop:
  1. **Submit phase**: build all `generate_mutation` prompts for the batch up front (this needs each prompt's parent-selection/crossover logic to run first, synchronously, since prompts depend on tournament/diverse-parent selection against the DB -- that part doesn't change), submit them as a single Message Batch job with the model's structured-output schema per request, and persist the batch job id.
  2. **Poll phase**: a separate loop (driven by the headless script, not by `run_generation_cycle`'s current `while` loop) polls the batch job until complete, then maps each result back to its originating candidate by `custom_id`, runs the existing local pre-filter (`EvolutionEngine._prefilter_reject`) on each, and submits a *second* batch for `critique_mutation` on whatever survives the pre-filter -- then polls that one too.
  3. Only after both batches resolve does the generation's survivor list exist, and `on_survivor` persistence (already incremental/durable per survivor, see engine.py's comment on that callback) runs against the batch results instead of per-call results.

This means `_create_candidate`'s current one-candidate-at-a-time shape doesn't
carry over unchanged into batch mode; a batch-mode headless runner would look
structurally different from today's `EvolutionEngine.run_generation_cycle`
caller, closer to "submit generation N's prompts, await both batches, then
call the same survivor-persistence code" than to the current tight per-candidate
async loop. `EvolutionEngine`'s DB-facing logic (parent selection, adaptive
threshold, survivor persistence) is reusable as-is; only the LLM-calling shape
changes.

## What would need to change

- A new provider entry in the pool shape, or more likely a separate code path
  entirely (`LLMEngine` methods return one result per call today; a batch
  provider fundamentally returns N results after a poll, which doesn't fit the
  `_try_provider(...) -> T` signature without either faking synchronicity with
  a blocking poll per call -- defeating the purpose -- or exposing a distinct
  `submit_batch(prompts) -> batch_id` / `poll_batch(batch_id) -> dict[custom_id, T]`
  pair that headless scripts call directly instead of `generate_mutation`/`critique_mutation`.
- `EvolutionEngine` would need a batch-aware variant of `run_generation_cycle`
  (or a parameter/flag) that separates "decide what to generate" from "await
  the LLM," since today those are the same awaited call.
- Cost/quota accounting (`LLMStats`) would need batch-specific fields --
  Anthropic's Batches API is billed at a discount versus synchronous calls, and
  that's part of the motivation for wanting this at all for large unattended
  runs.
- Error handling: a batch job can partially fail (some `custom_id`s succeed,
  others error) in ways the current per-call retry/fallback logic in
  `LLMEngine._completion`/`_try_provider` doesn't model at all.

## Why opt-in and headless-only

- The interactive menu's UX contract (a spinner, then results to score in the
  same session) is incompatible with minutes-to-24h turnaround; making batch
  mode the default would silently break that contract for anyone who didn't
  ask for it.
- Headless scripts already opt into non-default behavior per-script (see
  `headless_evolve_rule30.py` pinning `max_concurrent_calls` and swapping in a
  single explicit provider) -- an opt-in batch mode fits that same pattern:
  a script explicitly requests it, rather than `LLMEngine` silently switching
  behavior in ways that would be confusing to reason about from `cli.py`'s
  side.

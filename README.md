# pocketfm-world-models

`pocketfm-world-models` is a multi-agent story simulation for generating character-driven narratives with an LLM-backed world model. It can run entirely through Gemini, or use a locally trained DPO action adapter for character action generation while Gemini continues to handle criticism, narration, and goal progression.

The loop is:

- each character proposes one concrete next action
- a critic checks whether the action is plausible, in-character, non-repetitive, and goal-relevant
- a narrator chooses which proposed actions become canonical story events
- the environment applies selected `world_updates` to a graph-backed world state
- optional RAG memory stores narrated beats and retrieves relevant prior context

## Current Behavior

The default CLI entrypoint is [`run_pipeline.py`](/Users/chloeho/Documents/pocketfm/pocketfm-world-models/run_pipeline.py).

Default runtime values:

- story: `radio`, which maps to `radio_station_last_show`
- max steps: `30`
- max plan revisions: `1`
- RAG: disabled unless `--use-rag` or `USE_RAG=1` is set
- RAG retrieval count: `2`

Gemini is always required for critic, narrator, and follow-up goal generation. If `ACTION_MODEL_PATH` is set, character action generation uses the local DPO adapter loaded by `ActionAdapterLLM`; otherwise it uses Gemini too.

## Architecture

- [`run_pipeline.py`](/Users/chloeho/Documents/pocketfm/pocketfm-world-models/run_pipeline.py): CLI entrypoint, story selection, model wiring, graph export, and final story logging
- [`pipeline.py`](/Users/chloeho/Documents/pocketfm/pocketfm-world-models/pipeline.py): simulation loop, optional memory use, narrator selection, and goal progression
- [`agents.py`](/Users/chloeho/Documents/pocketfm/pocketfm-world-models/agents.py): `CharacterAgent` and `NarratorAgent`
- [`environment.py`](/Users/chloeho/Documents/pocketfm/pocketfm-world-models/environment.py): `networkx.DiGraph` world state and protected state updates
- [`fables.py`](/Users/chloeho/Documents/pocketfm/pocketfm-world-models/fables.py): built-in story definitions and aliases
- [`prompts.py`](/Users/chloeho/Documents/pocketfm/pocketfm-world-models/prompts.py): action, critic, narrator, and next-goal prompts
- [`llm.py`](/Users/chloeho/Documents/pocketfm/pocketfm-world-models/llm.py): Gemini client plus local DPO action-adapter inference
- [`memory.py`](/Users/chloeho/Documents/pocketfm/pocketfm-world-models/memory.py): optional structured memory with embedding retrieval
- [`generate_dpo_dataset.py`](/Users/chloeho/Documents/pocketfm/pocketfm-world-models/generate_dpo_dataset.py): preference dataset generation from story rollouts
- [`train_dpo.py`](/Users/chloeho/Documents/pocketfm/pocketfm-world-models/train_dpo.py): LoRA DPO adapter training
- [`sim_types.py`](/Users/chloeho/Documents/pocketfm/pocketfm-world-models/sim_types.py): shared dataclasses

## Built-In Stories

Supported names and aliases are registered in `get_fable_definition()`.

- `maya story`: corner-store romance, internally `corner_store_last_week`
- `wedding_weekend`: family wedding pressure story
- `restaurant_last_service`: final dinner-service ensemble story
- `flood_rescue`: floodwater rescue story, internally `flood_rescue_night`
- `radio`: final radio broadcast story, internally `radio_station_last_show`
- `codicil`: probate and missing-will story, internally `the_missing_codicil`

Example:

```bash
python run_pipeline.py --story wedding_weekend --steps 20
```

## Setup

```bash
pip install -r requirements.txt
```

Core dependencies include:

- `networkx`
- `llama-index-core`
- `llama-index-embeddings-huggingface`
- `torch`
- `transformers`
- `peft`
- `trl`
- `datasets`
- `bitsandbytes`

Some DPO and local-adapter paths need Hugging Face model access and suitable local compute. GPU is strongly preferred for training.

## Environment Variables

General runtime:

- `GEMINI_API_KEY`: required
- `GEMINI_MODEL`: optional, defaults to `gemini-2.5-flash`
- `GEMINI_BASE_URL`: optional, defaults to `https://generativelanguage.googleapis.com/v1beta`
- `GEMINI_OUTPUT_LOG_PATH`: optional, defaults to `llm_output.txt`
- `FABLE_NAME`: optional fallback story name for `run_pipeline.py`
- `MAX_STEPS`: optional fallback for `--steps`, defaults to `30`
- `MAX_PLAN_REVISIONS`: optional fallback for `--max-plan-revisions`, defaults to `1`
- `USE_RAG`: optional, set to `1`, `true`, or `yes` to enable memory retrieval
- `RAG_K`: optional fallback for `--rag-k`, defaults to `2`
- `WORLD_GRAPH_OUTPUT_PATH`: optional, defaults to `final_world_graph.json`

Local DPO action adapter:

- `ACTION_MODEL_PATH`: path to a trained adapter; when set, character actions use the local adapter
- `ACTION_MODEL_BASE`: optional base model override if it cannot be resolved from `adapter_config.json`
- `ACTION_MODEL_MAX_NEW_TOKENS`: optional, defaults to `96`
- `ACTION_MODEL_TEMPERATURE`: optional, defaults to `0.3`
- `ACTION_MODEL_LOAD_IN_4BIT`: optional, set to `1`, `true`, or `yes` for 4-bit loading
- `ACTION_MODEL_OUTPUT_LOG_PATH`: optional, defaults to `GEMINI_OUTPUT_LOG_PATH` or `llm_output.txt`

## Running Stories

Run the default radio story with Gemini-only action generation:

```bash
export GEMINI_API_KEY=your_api_key_here
python run_pipeline.py
```

Run a specific story:

```bash
export GEMINI_API_KEY=your_api_key_here
python run_pipeline.py --story flood_rescue --steps 25
```

Enable memory retrieval:

```bash
export GEMINI_API_KEY=your_api_key_here
python run_pipeline.py --story radio --use-rag --rag-k 3
```

Run with a local DPO action adapter:

```bash
export GEMINI_API_KEY=your_api_key_here
export ACTION_MODEL_PATH=artifacts/gemma-action-dpo
python run_pipeline.py --story radio
```

## DPO Workflow

The DPO workflow has two stages: generate preference data, then train a LoRA adapter.

### 1. Generate Preference Data

[`generate_dpo_dataset.py`](/Users/chloeho/Documents/pocketfm/pocketfm-world-models/generate_dpo_dataset.py) rolls out story episodes, samples two candidate actions per character, asks a pairwise judge to choose the better action, critiques the chosen action for world updates, and writes JSONL rows.

Default output:

- `artifacts/dpo_preferences.jsonl`

Generate a fresh dataset:

```bash
export GEMINI_API_KEY=your_api_key_here
python generate_dpo_dataset.py --fable radio --episodes 20 --max-steps 8 --overwrite
```

Useful options:

- `--fable`: story name, defaults to `maya story`
- `--episodes`: rollout count, defaults to `10`
- `--max-steps`: maximum steps per rollout, defaults to `8`
- `--output`: JSONL output path, defaults to `artifacts/dpo_preferences.jsonl`
- `--overwrite`: replace the output file instead of appending
- `--rag-k`: memory snippets per decision, defaults to `0`
- `--temperature`: candidate action sampling temperature, defaults to `0.8`
- `--max-new-goals`: follow-up goals per episode, defaults to `1`
- `--seed`: candidate-order and retry seed, defaults to `42`

Each JSONL row includes:

- `episode`, `step`, and `character`
- `prompt` payload used to rebuild the action prompt
- `chosen` and `rejected` action strings
- `chosen_rationale` and `rejected_rationale`
- `judge` pairwise decision metadata
- `chosen_eval` critic output for the chosen action

### 2. Train A DPO Adapter

[`train_dpo.py`](/Users/chloeho/Documents/pocketfm/pocketfm-world-models/train_dpo.py) converts the generated JSONL rows into TRL DPO examples and trains a LoRA adapter.

Default input and output:

- input: `artifacts/dpo_preferences.jsonl`
- base model: `google/gemma-2-2b-it`
- output adapter: `artifacts/gemma-action-dpo`

Train with defaults:

```bash
python train_dpo.py
```

Train in 4-bit mode:

```bash
python train_dpo.py --load-in-4bit
```

Useful options:

- `--dataset`: JSONL preference dataset path
- `--model`: Hugging Face base model, defaults to `google/gemma-2-2b-it`
- `--output`: adapter output directory
- `--max-seq-length`: context length, defaults to `2048`
- `--batch-size`: per-device train batch size, defaults to `2`
- `--grad-accum`: gradient accumulation steps, defaults to `4`
- `--epochs`: training epochs, defaults to `1.0`
- `--lr`: learning rate, defaults to `5e-5`
- `--beta`: DPO beta, defaults to `0.1`
- `--eval-fraction`: held-out eval fraction, defaults to `0.05`
- `--limit`: optional row cap for quick tests
- `--lora-rank`: LoRA rank, defaults to `16`
- `--lora-alpha`: LoRA alpha, defaults to `16`
- `--keep-rationales`: include rationales in chosen/rejected completions
- `--lora-target-modules`: `auto`, `attention-only`, or comma-separated module names

By default, training omits rationales from completions so the preference loss focuses on the action text rather than explanation style. If the dataset lacks explicit confidence fields, the trainer writes the same fallback confidence into chosen and rejected completions to avoid creating an artificial shortcut.

## How A Step Works

For each timestep:

1. the pipeline reads current world variables and the active goal
2. each character optionally retrieves memory snippets when RAG is enabled
3. the character agent builds an action prompt from persona, goal, curated world knowledge, recent scenes, memory snippets, and revision feedback when present
4. the action LLM proposes JSON with `action`, `confidence`, and `rationale`
5. the critic LLM evaluates the action and may request revision
6. the narrator LLM selects a small subset of proposed actions and writes one scene paragraph
7. selected actions are applied to the environment
8. optional memory records the narrated beat
9. if the current goal completes, the narrator LLM generates a follow-up goal
10. if a goal has not completed after 15 timesteps since it became active, the narrator LLM replaces it with a feasible goal based on the current story state

## World Model

[`environment.py`](/Users/chloeho/Documents/pocketfm/pocketfm-world-models/environment.py) stores the world in a directed graph with:

- one `world` node
- one node per character
- one node per world state key

Core world variables include:

- `turn`
- `characters`
- `fable_name`
- `current_goal`
- `goal_history`
- `goal_reached`

Critic-provided `world_updates` can add or change story state. Protected keys cannot be overwritten through critic updates:

- `turn`
- `characters`
- `fable_name`
- `current_goal`
- `goal_history`

## Output

`run_pipeline.py` prints:

- `done`
- `steps`
- `total_reward`
- `world_vars`
- `world_graph_path`
- the event timeline
- the final story paragraphs

Artifacts:

- `final_world_graph.json`: exported NetworkX node-link graph by default
- `llm_output.txt`: prompt/output log by default
- `artifacts/dpo_preferences.jsonl`: default generated DPO preference dataset
- `artifacts/gemma-action-dpo`: default trained DPO adapter directory

## Notes And Limitations

- Gemini access is required even when using a local action adapter, because critic, narrator, and next-goal calls still use Gemini.
- RAG memory is disabled by default and only initialized when requested.
- The embedding backend may download local model assets on first use.
- `StepResult.done` is not currently set to `True` by the environment, so runs usually stop by step limit rather than terminal state.
- `_should_act_now()` currently always returns `True`, so every character proposes an action on every timestep.
- DPO generation and training are experimental and depend heavily on the quality of pairwise preference judgments.

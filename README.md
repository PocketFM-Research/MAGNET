# pocketfm-world-models

`pocketfm-world-models` is a small multi-agent story simulation for generating character-driven narratives with an LLM-backed world model.

The codebase currently runs a loop where:

- each character proposes a concrete next action
- a critic checks whether that action is plausible and goal-relevant
- a narrator selects the canonical actions for the current beat and writes the prose
- the environment applies the selected state updates to a world graph
- memory stores prior beats and retrieves relevant context for later decisions

## Current Behavior

The repository currently supports two built-in story definitions:

- `ant_and_dove`
- `maya story`

The default entrypoint in [`run_pipeline.py`](/Users/chloeho/Documents/pocketfm/pocketfm-world-models/run_pipeline.py) now uses:

- `FABLE_NAME="maya story"` when unset
- `max_steps=15`
- `max_plan_revisions=1`
- `rag_k=2`

The default story is the corner-store romance scenario, not the ant-and-dove scenario.

## What The System Does

- simulates a story over multiple timesteps
- keeps world state in a `networkx.DiGraph`
- tracks structured episodic memory with embedding retrieval via `llama-index`
- calls Gemini in JSON mode for action generation, criticism, narration, and next-goal generation
- exports the final world graph to JSON
- appends prompts, outputs, and the final story to a log file

## Architecture

- [`run_pipeline.py`](/Users/chloeho/Documents/pocketfm/pocketfm-world-models/run_pipeline.py): entrypoint that loads a fable, runs the pipeline, prints output, and writes logs/artifacts
- [`pipeline.py`](/Users/chloeho/Documents/pocketfm/pocketfm-world-models/pipeline.py): orchestration loop for agents, narrator selection, goal progression, and memory writes
- [`agents.py`](/Users/chloeho/Documents/pocketfm/pocketfm-world-models/agents.py): `CharacterAgent` and `NarratorAgent`
- [`environment.py`](/Users/chloeho/Documents/pocketfm/pocketfm-world-models/environment.py): graph-backed world environment and state update logic
- [`fables.py`](/Users/chloeho/Documents/pocketfm/pocketfm-world-models/fables.py): `FableDefinition` plus built-in story setups
- [`memory.py`](/Users/chloeho/Documents/pocketfm/pocketfm-world-models/memory.py): structured memory entries, vector index creation, and retrieval
- [`llm.py`](/Users/chloeho/Documents/pocketfm/pocketfm-world-models/llm.py): Gemini API wrapper
- [`prompts.py`](/Users/chloeho/Documents/pocketfm/pocketfm-world-models/prompts.py): prompt templates for action, critic, narrator, and next-goal stages
- [`sim_types.py`](/Users/chloeho/Documents/pocketfm/pocketfm-world-models/sim_types.py): shared dataclasses

## How A Step Works

For each timestep:

1. the pipeline reads the current world variables and active goal
2. each character retrieves relevant memory snippets
3. the character agent builds a compact prompt from persona, goal, curated world knowledge, the last two narrated scenes, and revision feedback when available
4. the LLM proposes an action
5. the critic decides whether that action should be revised and whether it advances or completes the goal
6. the narrator selects a small subset of proposed actions and turns them into a paragraph
7. only narrator-selected actions are applied to the environment
8. the resulting beat is stored in memory
9. if a selected action completes the goal, the narrator generates a follow-up goal

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

The critic can also propose additional `world_updates`. Reserved keys are protected and cannot be overwritten through critic updates:

- `turn`
- `characters`
- `fable_name`
- `current_goal`
- `goal_history`

## Built-In Stories

### `maya story`

This maps to the `corner_store_last_week` fable in [`fables.py`](/Users/chloeho/Documents/pocketfm/pocketfm-world-models/fables.py).

Premise:

`Maya must decide whether to risk a real romance with Omar before he gives an answer on a job offer that would take him away.`

Characters:

- `Maya`
- `Rafael`
- `Leah`
- `Omar`

Initial world state includes relationship tension, a job-offer deadline, and a neighborhood setting.

### `ant_and_dove`

Premise:

`The ant saves the dove from the hunter.`

Characters:

- `Ant`
- `Dove`
- `Spider`

## Requirements

- Python 3.10+
- a valid `GEMINI_API_KEY`

## Setup

```bash
pip install -r requirements.txt
```

## Environment Variables

- `GEMINI_API_KEY`: required
- `GEMINI_MODEL`: optional, defaults to `gemini-2.5-flash`
- `GEMINI_BASE_URL`: optional, defaults to `https://generativelanguage.googleapis.com/v1beta`
- `GEMINI_OUTPUT_LOG_PATH`: optional, defaults to `llm_output.txt`
- `FABLE_NAME`: optional, defaults to `maya story`
- `WORLD_GRAPH_OUTPUT_PATH`: optional, defaults to `final_world_graph.json`

## Running

Default run:

```bash
export GEMINI_API_KEY=your_api_key_here
python run_pipeline.py
```

Run the ant-and-dove scenario explicitly:

```bash
export GEMINI_API_KEY=your_api_key_here
export FABLE_NAME="ant_and_dove"
python run_pipeline.py
```

Run the current default romance scenario explicitly:

```bash
export GEMINI_API_KEY=your_api_key_here
export FABLE_NAME="maya story"
python run_pipeline.py
```

## Output

The program prints:

- a summary dict with `done`, `steps`, `total_reward`, `world_vars`, and `world_graph_path`
- a timeline of proposed actions, selected actions, narration, and canonical events
- the final story paragraphs

Artifacts written to disk:

- `final_world_graph.json` by default for the exported NetworkX node-link graph
- `llm_output.txt` by default for prompt/output logging

At the end of a run, the final story is appended to the log file as a separate block.



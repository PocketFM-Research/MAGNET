import argparse
import json
import os
from datetime import datetime, timezone
from networkx.readwrite import json_graph
from environment import WorldProxyEnv
from fables import get_fable_definition
from llm import build_action_llm, build_default_llm
from pipeline import Config, Pipeline

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the pocketfm world-model story pipeline.")
    parser.add_argument(
        "--story",
        "--fable",
        dest="fable_name",
        default=os.getenv("FABLE_NAME", "radio"),
        help="Built-in story/fable name to run. Falls back to FABLE_NAME if unset.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=int(os.getenv("MAX_STEPS", "30")),
        help="Maximum number of simulation steps. Falls back to MAX_STEPS or 30.",
    )
    parser.add_argument(
        "--max-plan-revisions",
        type=int,
        default=int(os.getenv("MAX_PLAN_REVISIONS", "1")),
        help="Maximum critic-requested action revisions per character step.",
    )
    parser.add_argument(
        "--use-rag",
        action="store_true",
        default=os.getenv("USE_RAG", "0").strip().lower() in {"1", "true", "yes"},
        help="Enable memory retrieval for character prompts.",
    )
    parser.add_argument(
        "--rag-k",
        type=int,
        default=int(os.getenv("RAG_K", "2")),
        help="Number of memory snippets to retrieve when RAG is enabled.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY is required.")

    fable = get_fable_definition(args.fable_name)
    characters = fable.characters

    critic_llm = build_default_llm()
    action_model_path = os.getenv("ACTION_MODEL_PATH", "").strip()
    if action_model_path:
        action_llm = build_action_llm()
        pipeline = Pipeline(llm=critic_llm, action_llm=action_llm, use_rag=args.use_rag)
    else:
        pipeline = Pipeline(llm=critic_llm, use_rag=args.use_rag)
    env = WorldProxyEnv(fable=fable)
    result = pipeline.run(
        env=env,
        characters=characters,
        cfg=Config(
            goal=fable.goal,
            max_steps=args.steps,
            max_plan_revisions=args.max_plan_revisions,
            use_rag=args.use_rag,
            rag_k=args.rag_k,
        ),
    )

    graph_path = os.getenv("WORLD_GRAPH_OUTPUT_PATH", "final_world_graph.json")
    graph_exported = False
    try:
        try:
            graph_data = json_graph.node_link_data(env.world_graph, edges="links")
        except TypeError:
            graph_data = json_graph.node_link_data(env.world_graph, link="links")
        with open(graph_path, "w", encoding="utf-8") as handle:
            json.dump(graph_data, handle, indent=2)
            handle.write("\n")
        graph_exported = True
    except OSError:
        pass

    print(
        {
            "done": result["done"],
            "steps": result["steps"],
            "total_reward": result["total_reward"],
            "world_vars": result["world_vars"],
            "world_graph_path": graph_path if graph_exported else None,
        }
    )
    print("--- timeline ---")
    for line in result["timeline"]:
        print(line)
    print("--- story ---")
    for paragraph in result["story"]:
        print(paragraph)

    log_path = os.getenv("GEMINI_OUTPUT_LOG_PATH", "llm_output.txt")
    try:
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(
                f"[{datetime.now(timezone.utc).isoformat()}] FINAL STORY\n"
                "=== STORY START ===\n"
            )
            for paragraph in result["story"]:
                handle.write(f"{paragraph}\n")
            handle.write("=== STORY END ===\n\n")
    except OSError:
        pass


if __name__ == "__main__":
    main()

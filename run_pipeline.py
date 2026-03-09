import os
from datetime import datetime, timezone
from environment import WorldProxyEnv
from fables import get_fable_definition
from pipeline import Config, Pipeline

def main() -> None:
    if not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY is required.")

    fable_name = os.getenv("FABLE_NAME", "ant_and_dove")
    fable = get_fable_definition(fable_name)
    characters = fable.characters

    pipeline = Pipeline()
    env = WorldProxyEnv(fable=fable)
    result = pipeline.run(
        env=env,
        characters=characters,
        cfg=Config(goal=fable.goal, max_steps=10, max_plan_revisions=1, rag_k=2),
    )

    print(
        {
            "done": result["done"],
            "steps": result["steps"],
            "total_reward": result["total_reward"],
            "world_vars": result["world_vars"],
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

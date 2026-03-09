import os
from datetime import datetime, timezone
from environment import WorldProxyEnv
from pipeline import Config, Pipeline
from sim_types import CharacterProfile

def main() -> None:
    if not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY is required.")

    characters = [
        CharacterProfile(
            name="Ant",
            role="helper",
            traits=["hardworking", "small", "persistent"],
            goals=["find food", "survive", "repay kindness"],
            fears=["drowning"],
            abilities=["bite", "crawl", "gather food"],
            relationships={"Dove": "grateful ally"},
            state={"injured": False, "wet": False},
            description="An observant ant who remembers favors and acts decisively under pressure.",
        ),
        CharacterProfile(
            name="Dove",
            role="protector",
            traits=["compassionate", "alert", "brave"],
            goals=["protect nearby creatures", "avoid the hunter"],
            fears=["hunter's arrows"],
            abilities=["fly", "spot danger", "carry twigs and leaves"],
            relationships={"Ant": "trusted friend"},
            state={"injured": False, "nest_safe": True},
            description="A watchful dove who intervenes quickly when others are in danger.",
        ),
    ]

    pipeline = Pipeline()
    env = WorldProxyEnv()
    result = pipeline.run(
        env=env,
        characters=characters,
        cfg=Config(goal="recreate_ant_and_dove_fable", max_steps=6, max_plan_revisions=1, rag_k=2),
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

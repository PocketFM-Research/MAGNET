import os
from environment import WorldProxyEnv
from pipeline import Config, Pipeline
from sim_types import CharacterProfile

def main() -> None:
    if not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY is required.")

    characters = [
        CharacterProfile(
            name="Ant",
            description=(
                "A diligent ant who gathers food and learns from danger. "
                "After being rescued, the ant repays kindness."
            ),
        ),
        CharacterProfile(
            name="Dove",
            description=(
                "A compassionate dove who watches over others and acts quickly to help. "
                "After facing the hunter, the dove becomes more vigilant."
            ),
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


if __name__ == "__main__":
    main()

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sim_types import CharacterProfile


@dataclass
class FableDefinition:
    name: str
    goal: str
    characters: list[CharacterProfile]
    initial_world_vars: dict[str, Any]
    progress_reward: float = 0.5
    fallback_reward: float = 0.0
    completion_key: str = "goal_reached"

def define_corner_store_fable() -> FableDefinition:
    characters = [
        CharacterProfile(
            name="Maya",
            role="shop owner",
            traits=["guarded", "funny", "responsible"],
            goals=["protect her routines", "be honest about what she wants", "avoid another failed relationship"],
            fears=["misreading Omar", "upending her life for the wrong person"],
            abilities=["read people well", "keep conversations gentle", "notice small changes in others"],
            relationships={
                "Rafael": "father who wants her to stop hiding behind work",
                "Leah": "best friend who sees through her defenses",
                "Omar": "longtime neighbor she may already be in love with",
            },
            state={"at_store": True, "guarded": True, "hopeful": False},
            description=(
                "A 29-year-old woman who has spent so long keeping the corner store and her family together "
                "that she no longer trusts herself to want anything more personal."
            ),
        ),
        CharacterProfile(
            name="Rafael",
            role="father",
            traits=["warm", "perceptive", "old-fashioned"],
            goals=["see Maya build a life beyond work", "keep family ties honest", "nudge without controlling"],
            fears=["watching Maya settle for loneliness", "saying too much too soon"],
            abilities=["put people at ease", "notice emotional subtext", "offer advice without preaching"],
            relationships={
                "Maya": "daughter he loves and worries is holding back",
                "Leah": "trusted family friend",
                "Omar": "young man he quietly approves of",
            },
            state={"at_store": True, "watchful": True},
            description=(
                "A widower in his sixties who has learned that love often shows up as patience, timing, and "
                "the courage to say one truthful thing at the right moment."
            ),
        ),
        CharacterProfile(
            name="Leah",
            role="friend",
            traits=["witty", "direct", "loyal"],
            goals=["push Maya to be honest", "keep the friendship balanced", "prevent regret from masquerading as caution"],
            fears=["watching Maya miss her chance", "becoming the friend who meddles too hard"],
            abilities=["ask hard questions kindly", "lighten tense moments", "spot mutual attraction quickly"],
            relationships={
                "Maya": "best friend who needs occasional shoving",
                "Rafael": "father figure she respects",
                "Omar": "friend she trusts but teases relentlessly",
            },
            state={"available": True, "suspicious_of_chemistry": True},
            description=(
                "A public school counselor who has known both Maya and Omar for years and is tired of watching "
                "them confuse caution with maturity."
            ),
        ),
        CharacterProfile(
            name="Omar",
            role="neighbor",
            traits=["steady", "reserved", "deeply reliable"],
            goals=["tell Maya how he feels", "make the right choice about a job offer", "avoid pressuring her"],
            fears=["damaging their friendship", "leaving without saying what mattered"],
            abilities=["listen carefully", "show care through practical help", "stay calm under emotional pressure"],
            relationships={
                "Maya": "neighbor and closest friend he has quietly loved for years",
                "Rafael": "older mentor",
                "Leah": "friend who knows more than he says aloud",
            },
            state={"available": True, "job_offer_pending": True, "ready_to_leave": False},
            description=(
                "A building superintendent from two doors down who rarely speaks dramatically, but whose care "
                "for Maya has become impossible to hide now that he may leave the neighborhood."
            ),
        ),
    ]

    return FableDefinition(
        name="corner_store_last_week",
        goal="Maya must decide whether to risk a real romance with Omar before he gives an answer on a job offer that would take him away.",
        characters=characters,
        initial_world_vars={
            "goal_reached": False,
            "setting": "A neighborhood corner store and the apartments above it during one emotionally tense week.",
            "job_offer_deadline_days": 6, 
            "shared_history": "Maya and Omar have spent years circling around feelings neither of them has named plainly.",
            "current_tension": "Omar has a stable out-of-state job offer, and Maya has just realized losing him would feel personal, not merely practical.",
            "community_mood": "close-knit, observant, and aware that something unspoken is finally coming to a head",
        },
        progress_reward=0.5,
        fallback_reward=0.0,
        completion_key="goal_reached",
    )


def define_wedding_weekend_fable() -> FableDefinition:
    characters = [
        CharacterProfile(
            name="Nina",
            role="older sister",
            traits=["competent", "wound-tight", "loving"],
            goals=["get through the wedding weekend without a family blowup", "protect her younger brother from humiliation", "stop pretending she is not exhausted"],
            fears=["becoming her mother under stress", "letting resentment ruin the weekend"],
            abilities=["organize chaos", "read family tension quickly", "keep appearances intact"],
            relationships={
                "Eli": "younger brother she has covered for since childhood",
                "Sofia": "future sister-in-law she genuinely likes",
                "Marianne": "mother whose approval still gets under her skin",
            },
            state={"at_venue": True, "holding_it_together": True},
            description=(
                "A high-functioning eldest daughter who can manage a room full of relatives, but is no longer sure "
                "she can manage her own anger."
            ),
        ),
        CharacterProfile(
            name="Eli",
            role="groom",
            traits=["charming", "avoidant", "kind"],
            goals=["marry Sofia without hurting anyone", "keep peace between his mother and sister", "finally act like an adult under pressure"],
            fears=["freezing when conflict gets direct", "starting his marriage by disappointing Sofia"],
            abilities=["defuse tension with humor", "admit fault when cornered", "make people feel chosen"],
            relationships={
                "Nina": "sister who has always rescued him",
                "Sofia": "fiancee he deeply loves",
                "Marianne": "mother whose moods he still tiptoes around",
            },
            state={"wedding_weekend": True, "conflict_avoidant": True},
            description=(
                "A man who has spent years surviving family tension by charming his way around it, and is about to learn "
                "that marriage requires a firmer spine."
            ),
        ),
        CharacterProfile(
            name="Sofia",
            role="bride",
            traits=["warm", "observant", "self-respecting"],
            goals=["protect the joy of the weekend", "understand what kind of family she is marrying into", "force honesty before the ceremony"],
            fears=["mistaking politeness for stability", "being welcomed only if she stays quiet"],
            abilities=["stay calm in awkward rooms", "ask direct questions gently", "notice hidden loyalties"],
            relationships={
                "Eli": "fiance she loves but needs to trust more fully",
                "Nina": "future sister-in-law she suspects is carrying too much",
                "Marianne": "future mother-in-law she is trying to read correctly",
            },
            state={"ready_to_marry": True, "watchful": True},
            description=(
                "A public-interest lawyer who knows how to stay gracious under pressure, but refuses to build a life on top of denied truths."
            ),
        ),
        CharacterProfile(
            name="Marianne",
            role="mother",
            traits=["elegant", "controlling", "easily wounded"],
            goals=["keep authority over the family narrative", "avoid being sidelined at the wedding", "force Nina and Eli back into old roles"],
            fears=["irrelevance", "public embarrassment", "being seen clearly by Sofia"],
            abilities=["weaponize concern", "shift blame subtly", "turn tradition into leverage"],
            relationships={
                "Nina": "daughter who resists her most openly",
                "Eli": "son she still sees as emotionally dependent",
                "Sofia": "newcomer she is not sure she can control",
            },
            state={"dressed_for_rehearsal": True, "offended": False},
            description=(
                "A mother who believes she is holding the family together, even as everyone else quietly experiences her as the force they are trying to survive."
            ),
        ),
    ]

    return FableDefinition(
        name="wedding_weekend",
        goal="Before the wedding ceremony begins, Eli must stop avoiding the truth and choose whether he will protect Sofia and Nina from Marianne's control or let the old family pattern ruin the marriage before it starts.",
        characters=characters,
        initial_world_vars={
            "goal_reached": False,
            "setting": "A lakeside hotel hosting a wedding weekend full of old family habits and expensive politeness.",
            "current_event": "The rehearsal dinner is hours away, and everyone is pretending nothing is wrong.",
            "hidden_conflict": "Marianne has been quietly changing plans and pitting Nina and Eli against each other in the name of keeping the wedding elegant.",
            "time_pressure": "If the real conflict is not named before the ceremony, it will explode in front of both families.",
            "family_history": "Nina has spent years absorbing the emotional cost of Marianne's behavior while Eli smoothed things over and moved on.",
        },
        progress_reward=0.5,
        fallback_reward=0.0,
        completion_key="goal_reached",
    )


def define_restaurant_last_service_fable() -> FableDefinition:
    characters = [
        CharacterProfile(
            name="Theo",
            role="chef-owner",
            traits=["demanding", "talented", "burned out"],
            goals=["survive the restaurant's final service with dignity", "admit what the place has cost him", "stop mistaking control for care"],
            fears=["failing publicly in front of his staff", "being ordinary without the restaurant"],
            abilities=["lead a kitchen in crisis", "taste tiny problems immediately", "inspire loyalty through standards"],
            relationships={
                "Mara": "ex-wife and pastry chef he never fully let go of",
                "Luis": "sous-chef he trained but never properly trusted",
                "Jade": "server who sees the emotional weather of the room before anyone else",
            },
            state={"on_line": True, "closing_night": True},
            description=(
                "A chef whose identity fused with his restaurant so completely that losing it feels less like a business failure than a personal extinction."
            ),
        ),
        CharacterProfile(
            name="Mara",
            role="pastry chef",
            traits=["precise", "guarded", "unsentimental"],
            goals=["get through the last service without reopening the marriage", "leave the restaurant on her own terms", "protect Luis from Theo's worst habits"],
            fears=["being pulled back into Theo's chaos", "letting nostalgia rewrite the truth"],
            abilities=["stay composed under pressure", "finish work beautifully even when angry", "separate sentiment from reality"],
            relationships={
                "Theo": "ex-husband and creative partner she still understands too well",
                "Luis": "cook she mentors quietly",
                "Jade": "friend who knows the real history",
            },
            state={"scheduled_to_leave": True, "emotionally_armored": True},
            description=(
                "A pastry chef who learned long ago that beauty can be made in difficult places, but not every difficult place deserves to be saved."
            ),
        ),
        CharacterProfile(
            name="Luis",
            role="sous-chef",
            traits=["ambitious", "loyal", "underestimated"],
            goals=["prove he can run service when Theo cracks", "decide whether to inherit the kitchen's culture or reject it", "keep the staff from falling apart tonight"],
            fears=["becoming a smaller copy of Theo", "freezing when leadership is finally his"],
            abilities=["run expo efficiently", "steady younger cooks", "translate chaos into tasks"],
            relationships={
                "Theo": "mentor he admires and resents",
                "Mara": "the one person who tells him the truth",
                "Jade": "friend who reminds him the dining room is part of the story",
            },
            state={"on_expo": True, "ready_for_more": False},
            description=(
                "A sous-chef talented enough to take over, if he can stop waiting for permission from the man who taught him to doubt himself."
            ),
        ),
        CharacterProfile(
            name="Jade",
            role="lead server",
            traits=["sharp", "funny", "emotionally perceptive"],
            goals=["hold the front of house together", "make the final service feel worth remembering", "force the kitchen to stop treating collapse as romance"],
            fears=["watching everyone sentimentalize a toxic place", "letting the night become ugly for the staff or guests"],
            abilities=["read tables instantly", "manage staff morale", "say hard truths without flinching"],
            relationships={
                "Theo": "boss she respects but no longer indulges",
                "Mara": "friend she trusts",
                "Luis": "co-conspirator in keeping the place humane",
            },
            state={"floor_open": True, "tips_matter_tonight": True},
            description=(
                "A front-of-house leader who understands that the last night of a restaurant is never only about food; it is also about who gets to walk away intact."
            ),
        ),
    ]

    return FableDefinition(
        name="restaurant_last_service",
        goal="During the restaurant's final dinner service, Theo must decide whether to cling to control or hand real trust to Mara and Luis before the night collapses into one last beautiful disaster.",
        characters=characters,
        initial_world_vars={
            "goal_reached": False,
            "setting": "A once-beloved neighborhood restaurant on its final night of service, packed with regulars, old critics, and people hoping for one last perfect meal.",
            "current_service_state": "The dining room is full, the kitchen is short one line cook, and every mistake feels loaded with history.",
            "money_pressure": "The restaurant is closing after months of debt, and everyone knows there is no tomorrow to fix a bad night.",
            "shared_history": "Theo and Mara built the restaurant together, then ruined their marriage inside it without ever fully leaving each other.",
            "staff_mood": "Proud, sentimental, brittle, and trying not to show how badly they need the night to mean something.",
        },
        progress_reward=0.5,
        fallback_reward=0.0,
        completion_key="goal_reached",
    )


def define_flood_rescue_fable() -> FableDefinition:
    characters = [
        CharacterProfile(
            name="Elena",
            role="fire captain",
            traits=["decisive", "calm", "protective"],
            goals=["get every passenger out of the stranded bus alive", "build a stable rescue line before the water rises higher", "make fast choices without wasting effort"],
            fears=["losing civilians to hesitation", "sending her crew into water she cannot control"],
            abilities=["lead rescues", "size up flood hazards", "coordinate under pressure"],
            relationships={
                "Nico": "trusted rescue swimmer",
                "Priya": "dispatcher she relies on for timing and updates",
                "Marcus": "civilian bus driver she needs to keep focused",
            },
            state={"on_scene": True, "injured": False, "commanding": True},
            description=(
                "A fire captain who has handled flood calls before and knows that the first clear plan often decides who lives."
            ),
        ),
        CharacterProfile(
            name="Nico",
            role="rescue swimmer",
            traits=["brave", "fast", "practical"],
            goals=["reach the stranded bus before the water rises into the cabin", "help passengers cross the rescue line fast", "support Elena's plan"],
            fears=["being pinned by debris", "losing footing in the current"],
            abilities=["swim strong currents", "handle ropes", "carry injured people through water"],
            relationships={
                "Elena": "captain he trusts completely",
                "Priya": "voice in his earpiece keeping him aligned",
                "Marcus": "civilian he may need to physically extract",
            },
            state={"suited_up": True, "ready": True},
            description=(
                "A rescue specialist who works best in cold water, bad light, and the kind of panic that makes everyone else slower."
            ),
        ),
        CharacterProfile(
            name="Priya",
            role="dispatcher",
            traits=["focused", "precise", "steady"],
            goals=["track flood changes", "route support efficiently", "keep the field team ahead of the next surge"],
            fears=["missing a critical update", "giving stale information"],
            abilities=["monitor emergency channels", "map routes quickly", "spot timing windows"],
            relationships={
                "Elena": "field commander she feeds information to",
                "Nico": "rescuer she keeps supplied with updates",
                "Marcus": "civilian caller she is trying to stabilize",
            },
            state={"at_command_post": True, "comms_clear": True},
            description=(
                "An emergency dispatcher who turns scattered reports, weather updates, and radio traffic into a rescue clock everyone else can act on."
            ),
        ),
        CharacterProfile(
            name="Marcus",
            role="bus driver",
            traits=["responsible", "shaken", "selfless"],
            goals=["keep the children on his bus safe", "empty the bus seat by seat without panic", "stay behind until the last passenger is out"],
            fears=["the water breaking through the door seals", "leaving a child behind"],
            abilities=["move people in orderly lines", "keep frightened passengers focused", "follow instructions when they are clear"],
            relationships={
                "Elena": "rescuer he is trying to trust",
                "Nico": "last line between the passengers and the river",
                "Priya": "voice on the radio guiding him through each minute",
            },
            state={"stranded": True, "injured": False, "responsible_for_children": True},
            description=(
                "A city bus driver stranded with children aboard after floodwater trapped his route at a low-water crossing, trying to sound steadier than he feels."
            ),
        ),
    ]

    return FableDefinition(
        name="flood_rescue_night",
        goal="Elena and her team must evacuate passengers from a school bus stranded in floodwater before Marcus sacrifices himself to save the last children aboard and bring everyone to a local safety shelter.",
        characters=characters,
        initial_world_vars={
            "goal_reached": False,
            "setting": "A school bus is stalled at a low-water crossing outside town after flash floodwater washes over the road.",
            "rescue_layout": "The bus is angled in fast-moving water near a washed-out shoulder. Rescuers must anchor a rope line from dry ground to the bus and move passengers out one at a time.",
            "road_state": "The asphalt beyond the bus has started to crumble where the shoulder gave way, so no one can simply drive or walk the bus out.",
            "bus_state": "A school bus with children on board is trapped in knee- to waist-high floodwater, with one door jammed and water beginning to seep into the first step.",
            "weather": "Heavy rain, poor visibility, and rising water.",
            "time_pressure": "Upstream runoff is still arriving, and another surge could lift or roll the bus within minutes.",
            "fatal_turn": "Marcus is the last adult inside the bus and may need to brace the rear emergency exit long enough for the last children to escape, knowing the current could take him when the bus shifts.",
        },
        progress_reward=0.5,
        fallback_reward=0.0,
        completion_key="goal_reached",
    )


def define_radio_station_fable() -> FableDefinition:
    characters = [
        CharacterProfile(
            name="Camila",
            role="radio host",
            traits=["magnetic", "restless", "morally stubborn"],
            goals=["save the station's final live show from becoming a lie", "decide whether to expose the sponsor scandal on air", "protect the crew she helped build"],
            fears=["selling out the listeners who trusted her", "dragging everyone else down with her principles"],
            abilities=["hold a live room together", "draw truth out of callers", "improvise under pressure"],
            relationships={
                "Devin": "producer who knows how much chaos she can create and how often she is right",
                "Tessa": "station owner's daughter and journalist who has brought her dangerous documents",
                "Malik": "former co-host and local musician who still knows how to steady her",
            },
            state={"on_air_tonight": True, "contract_expiring": True, "angry": True},
            description=(
                "A beloved late-night radio host whose voice has kept lonely people company for years, now facing the last broadcast of the station she turned into a small-city institution."
            ),
        ),
        CharacterProfile(
            name="Devin",
            role="producer",
            traits=["careful", "funny", "exhausted"],
            goals=["get the show to air without the station collapsing in real time", "protect the staff from legal fallout", "keep Camila from making a reckless decision she cannot undo"],
            fears=["being the one who let everyone lose their jobs", "mistaking caution for cowardice"],
            abilities=["run live broadcasts cleanly", "triage crises fast", "read technical and human problems at once"],
            relationships={
                "Camila": "host he has protected and argued with for six years",
                "Tessa": "source he half-trusts and half-fears",
                "Malik": "friend from the station's earlier, scrappier days",
            },
            state={"in_control_room": True, "protective": True, "sleep_deprived": True},
            description=(
                "A producer who can patch dead air, calm panicked interns, and keep a show moving, but is no longer sure professionalism can solve a moral crisis."
            ),
        ),
        CharacterProfile(
            name="Tessa",
            role="investigative reporter",
            traits=["sharp", "contained", "unforgiving"],
            goals=["prove the station's main sponsor buried a toxic leak in the neighborhood by the river", "force her father to stop trading community trust for survival", "put the evidence somewhere it cannot be buried again"],
            fears=["becoming only the daughter of the man who covered it up", "hesitating long enough for the story to disappear"],
            abilities=["verify documents", "spot weak denials", "stay calm in hostile rooms"],
            relationships={
                "Camila": "public voice she believes might still tell the truth",
                "Devin": "gatekeeper she needs to win over",
                "Malik": "old friend who knows the city and the station better than either of them admits",
            },
            state={"carrying_documents": True, "disillusioned": True, "determined": True},
            description=(
                "The station owner's daughter, now a local reporter, who has spent years learning how to separate family loyalty from the facts and has finally run out of reasons to keep them separate."
            ),
        ),
        CharacterProfile(
            name="Malik",
            role="musician",
            traits=["warm", "observant", "lightly wounded"],
            goals=["help Camila choose a future that does not destroy her", "decide whether to play the farewell set or walk away from the station for good", "keep the night's truth from becoming just another performance"],
            fears=["being useful only as nostalgia", "watching Camila confuse sacrifice with courage"],
            abilities=["read crowds", "defuse tension", "say painful truths gently"],
            relationships={
                "Camila": "former co-host and almost-love he never fully got over",
                "Devin": "friend who stayed when he left",
                "Tessa": "ally of convenience whose evidence he knows could blow everything open",
            },
            state={"booked_to_perform": True, "still_in_love": True, "guarded": True},
            description=(
                "A once-promising local musician who owes much of his audience to the station and has returned for its final night knowing the city listens hardest when people are about to lose something."
            ),
        ),
    ]

    return FableDefinition(
        name="radio_station_last_show",
        goal="Before the final live broadcast ends, Camila must decide whether to expose a sponsor cover-up on air and risk the livelihoods of everyone at the station or protect the people she loves by letting the truth die with the show.",
        characters=characters,
        initial_world_vars={
            "goal_reached": False,
            "setting": "An aging independent radio station in a river city on the night before corporate ownership takes over.",
            "station_status": "The station is understaffed, emotionally frayed, and running a farewell broadcast that listeners across the city are treating like a wake.",
            "hidden_scandal": "Documents suggest the station's biggest sponsor concealed chemical contamination near a working-class neighborhood by the river, and station ownership knew enough to stay quiet.",
            "time_pressure": "Once the final hour ends, the control room archives will be locked, the sponsor lawyers will move, and the story may become impossible to prove publicly.",
            "public_mood": "The city is sentimental about the station and newly suspicious about what powerful people have asked it not to say.",
        },
        progress_reward=0.5,
        fallback_reward=0.0,
        completion_key="goal_reached",
    )


def define_probate_fable() -> FableDefinition:
    characters = [
        CharacterProfile(
            name="Asha",
            role="estate attorney",
            traits=["controlled", "empathetic", "tenacious"],
            goals=["settle the Serrano estate without letting the family destroy itself", "find out whether a missing codicil changed who inherits the house", "keep her own judgment clear around people she is beginning to care about"],
            fears=["missing a legal detail that cannot be undone", "confusing compassion with neutrality"],
            abilities=["read motives through paperwork", "manage conflict in formal settings", "notice inconsistencies others dismiss"],
            relationships={
                "Daniel": "older son whose certainty she does not trust",
                "Lucia": "younger daughter who may know more than she admits",
                "Rosa": "housekeeper and witness who has seen the family more clearly than they see themselves",
            },
            state={"handling_estate": True, "professionally_guarded": True, "under_pressure": True},
            description=(
                "A probate attorney known for keeping volatile families civil long enough to tell the truth, now assigned to a case that feels less like paperwork than a controlled detonation."
            ),
        ),
        CharacterProfile(
            name="Daniel",
            role="older son",
            traits=["capable", "resentful", "status-conscious"],
            goals=["sell the family house quickly", "prove he deserves control after years of carrying everyone else", "keep old humiliations from resurfacing in front of strangers"],
            fears=["discovering his mother trusted someone else more than him", "losing the house before he can convert it into cash and authority"],
            abilities=["take charge of practical tasks", "pressure people into decisions", "sound reasonable while hiding panic"],
            relationships={
                "Asha": "attorney he wants on his side",
                "Lucia": "sister whose moral certainty infuriates him",
                "Rosa": "employee he underestimates because she remembers too much",
            },
            state={"cash_strapped": True, "defensive": True, "living_in_house": False},
            description=(
                "The older child who has spent years believing responsibility should have bought him more love, and who intends to turn grief into an efficient transaction before anyone can stop him."
            ),
        ),
        CharacterProfile(
            name="Lucia",
            role="younger daughter",
            traits=["intuitive", "stubborn", "grieving"],
            goals=["stop the family house from being sold before the truth is known", "protect a version of her mother that was never visible in Daniel's story", "decide whether to reveal the letter she found hidden in the sewing room"],
            fears=["learning she misunderstood her mother too", "being dismissed as sentimental when she is actually right"],
            abilities=["remember emotional detail", "read silences in a room", "persist when everyone wants closure"],
            relationships={
                "Asha": "attorney she hopes is fair enough to listen",
                "Daniel": "brother whose urgency feels predatory to her",
                "Rosa": "surrogate aunt who helped raise her and holds dangerous context",
            },
            state={"still_in_house": True, "mourning": True, "holding_back_evidence": True},
            description=(
                "A documentary editor who returned home for her mother's funeral and stayed because the story her brother is trying to tell about the family feels too neat to be true."
            ),
        ),
        CharacterProfile(
            name="Rosa",
            role="housekeeper",
            traits=["practical", "devoted", "unshockable"],
            goals=["protect the house long enough for the right version of the will to surface", "force the Serrano children to see what their mother was trying to do", "stop being treated like furniture in the room where she knows the most"],
            fears=["watching love get rewritten into paperwork", "speaking too late to matter"],
            abilities=["notice physical details others miss", "keep routines alive under stress", "tell the truth without embellishment"],
            relationships={
                "Asha": "outsider she may be willing to trust",
                "Daniel": "boy she watched become a man without becoming kinder",
                "Lucia": "child she quietly protected for years",
            },
            state={"still_employed": False, "has_house_keys": True, "loyal_to_mother": True},
            description=(
                "The family's longtime housekeeper, executor's witness, and keeper of more history than any blood relative wants to admit, standing in the strange position of grieving and testifying at once."
            ),
        ),
    ]

    return FableDefinition(
        name="the_missing_codicil",
        goal="Before the Serrano estate closes, Asha must determine whether a missing codicil changed the inheritance of the family house and decide whom to believe before the sale turns a private betrayal into something permanent.",
        characters=characters,
        initial_world_vars={
            "goal_reached": False,
            "setting": "A once-grand row house in Philadelphia during the week after the matriarch's funeral, with probate deadlines closing in.",
            "estate_status": "The estate is in formal probate, the house has interested buyers, and every room still carries the habits of the woman who died there.",
            "missing_document": "A witness remembers the deceased amending her will months before her death, but the codicil is not in the file delivered to the attorney.",
            "financial_pressure": "Taxes, repairs, and private debts mean the estate cannot remain unresolved for long without costing everyone more than they can afford.",
            "family_secret": "The mother may have intended to leave the house to the person who stayed, rather than the child who always assumed it was his by right.",
        },
        progress_reward=0.5,
        fallback_reward=0.0,
        completion_key="goal_reached",
    )


def get_fable_definition(name: str) -> FableDefinition:
    key = name.lower().strip()
    if key in {"corner_store_last_week", "corner_store", "maya_story", "maya store", "maya story"}:
        return define_corner_store_fable()
    if key in {"wedding_weekend", "wedding_story", "family_wedding", "wedding"}:
        return define_wedding_weekend_fable()
    if key in {"restaurant_last_service", "restaurant_story", "closing_night", "restaurant"}:
        return define_restaurant_last_service_fable()
    if key in {"flood_rescue_night", "flood_rescue", "rescue_story", "action_story"}:
        return define_flood_rescue_fable()
    if key in {"radio_station_last_show", "radio_story", "last_show", "station_story", "radio"}:
        return define_radio_station_fable()
    if key in {"the_missing_codicil", "probate_story", "estate_story", "missing_will", "codicil"}:
        return define_probate_fable()
    raise ValueError(f"Unknown fable definition '{name}'.")

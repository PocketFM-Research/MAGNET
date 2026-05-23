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


def define_museum_restitution_fable() -> FableDefinition:
    characters = [
        CharacterProfile(
            name="Elena",
            role="museum curator",
            traits=["polished", "intelligent", "conflicted"],
            goals=["open the exhibition without scandal", "find out whether the museum knowingly accepted stolen art", "decide what kind of leader she wants to be"],
            fears=["destroying the institution she spent her life building", "mistaking loyalty for ethics"],
            abilities=["manage powerful people gracefully", "synthesize complex evidence quickly", "keep public panic contained"],
            relationships={
                "Jonah": "provenance researcher whose integrity she depends on",
                "Tariq": "journalist and former partner who still unsettles her",
                "Vivian": "board chair whose money and influence shape the museum",
            },
            state={"at_museum": True, "under_scrutiny": True, "public_face": True},
            description=(
                "A rising museum curator preparing for the biggest exhibition of her career, just as private doubts "
                "about one centerpiece object begin turning into a moral crisis she cannot delegate away."
            ),
        ),
        CharacterProfile(
            name="Jonah",
            role="provenance researcher",
            traits=["methodical", "earnest", "brave when cornered"],
            goals=["prove whether the ceremonial mask was looted during the civil war", "force the museum to act before the gala", "stop accepting institutional delay as neutrality"],
            fears=["being discredited before the truth is secure", "watching evidence disappear into committee language"],
            abilities=["trace archival gaps", "spot forged paperwork", "stay focused when others get political"],
            relationships={
                "Elena": "boss he respects but needs to pressure",
                "Tariq": "reporter he approached anonymously",
                "Vivian": "powerful board chair he suspects already knows enough",
            },
            state={"has_evidence": True, "ready_to_push": False, "professionally_exposed": False},
            description=(
                "A provenance specialist who has spent years in the footnotes of empire and now has one night to "
                "decide whether careful process is a virtue or a way of helping powerful people stall."
            ),
        ),
        CharacterProfile(
            name="Tariq",
            role="investigative journalist",
            traits=["probing", "composed", "unsentimental"],
            goals=["publish the truth with proof", "protect his source long enough to verify the story", "find out whether Elena will stand inside the institution or outside it"],
            fears=["running a story that can be dismissed as personal revenge", "watching Elena choose optics over courage"],
            abilities=["ask questions people cannot easily dodge", "build public pressure strategically", "read when silence means guilt"],
            relationships={
                "Elena": "former partner whose judgment still matters to him",
                "Jonah": "source he is trying to keep from being burned",
                "Vivian": "elite patron he has investigated before",
            },
            state={"off_the_record": True, "deadline_tonight": True, "emotionally_guarded": True},
            description=(
                "A reporter with a reputation for making elegant institutions explain themselves in plain language, "
                "now facing a story that is professionally explosive and personally unfinished."
            ),
        ),
        CharacterProfile(
            name="Vivian",
            role="board chair",
            traits=["formidable", "strategic", "ruthlessly calm"],
            goals=["save the gala, donors, and museum reputation", "contain any admission of wrongdoing", "keep Elena aligned with the board"],
            fears=["losing prestige through public contrition", "creating a precedent donors will interpret as weakness"],
            abilities=["apply pressure without raising her voice", "frame self-interest as stewardship", "make delay sound responsible"],
            relationships={
                "Elena": "protege she expects to act like an executive, not an activist",
                "Jonah": "staff researcher she considers naive and inconvenient",
                "Tariq": "reporter she views as opportunistic but dangerous",
            },
            state={"hosting_gala": True, "stonewalling": True, "publicly_confident": True},
            description=(
                "A board chair who genuinely believes institutions are preserved by controlling the timing of truth, "
                "even when that means deciding other people can wait longer for justice."
            ),
        ),
    ]

    return FableDefinition(
        name="museum_restitution_night",
        goal="Before the exhibition gala begins, Elena must decide whether to pull the museum's centerpiece mask, confront the board, and publicly acknowledge evidence that it was looted, or let the institution protect itself at the cost of the truth.",
        characters=characters,
        initial_world_vars={
            "goal_reached": False,
            "setting": "A major city museum during the final hours before a high-profile exhibition gala.",
            "featured_object": "A ceremonial mask on loan from a private collection, celebrated as the crown of the exhibition and now suspected to have been looted during wartime.",
            "institutional_pressure": "Donors, trustees, and the press are already arriving, and canceling the reveal could cost the museum millions and fracture the board.",
            "evidence_gap": "The archival chain of custody contains one missing year and a recently surfaced field report that directly contradicts the donor's paperwork.",
            "public_risk": "If the museum acts first, it may preserve credibility; if the story breaks externally, the institution will look like it hid what it knew.",
        },
        progress_reward=0.5,
        fallback_reward=0.0,
        completion_key="goal_reached",
    )


def define_hospital_night_shift_fable() -> FableDefinition:
    characters = [
        CharacterProfile(
            name="Camila",
            role="charge nurse",
            traits=["competent", "protective", "exhausted"],
            goals=["keep the emergency department functioning through the night", "protect her team from a preventable error", "stop carrying everyone else's fear alone"],
            fears=["losing a patient because she missed one small thing", "admitting how close she is to burnout"],
            abilities=["triage under pressure", "steady panicked people", "see the room as a system"],
            relationships={
                "Noah": "resident physician she pushes harder because he is capable of more",
                "Rina": "paramedic and closest friend on the worst nights",
                "Victor": "hospital administrator whose promises she no longer trusts",
            },
            state={"on_shift": True, "short_staffed": True, "holding_line": True},
            description=(
                "A charge nurse who can run an emergency department like a second nervous system, but who has reached "
                "the hour of the night where competence starts to feel indistinguishable from sacrifice."
            ),
        ),
        CharacterProfile(
            name="Noah",
            role="resident physician",
            traits=["brilliant", "insecure", "conscientious"],
            goals=["make the right call on a rapidly deteriorating patient", "earn Camila's trust", "admit a charting mistake before it harms someone"],
            fears=["being exposed as not ready for responsibility", "hesitating long enough for a patient to crash"],
            abilities=["diagnose unusual presentations", "learn quickly from correction", "stay with difficult cases longer than expected"],
            relationships={
                "Camila": "charge nurse whose respect matters more than he admits",
                "Rina": "paramedic who teases him but believes in him",
                "Victor": "administrator pushing throughput metrics over caution",
            },
            state={"covering_too_many_rooms": True, "made_documentation_error": True, "ready_to_confess": False},
            description=(
                "A gifted resident in the dangerous middle stage of training, where knowing a lot is not yet the same "
                "thing as trusting himself enough to speak up at the right moment."
            ),
        ),
        CharacterProfile(
            name="Rina",
            role="paramedic",
            traits=["fast", "funny", "unflinching"],
            goals=["get her crashing patient admitted before the system drops him", "force the staff to see a pattern in similar cases", "keep Camila from grinding herself into dust"],
            fears=["handing off a patient into institutional negligence", "becoming numb enough to stop caring"],
            abilities=["extract clear facts from chaos", "advocate fiercely during handoff", "notice when separate emergencies share one cause"],
            relationships={
                "Camila": "best friend whose limits she sees more clearly than Camila does",
                "Noah": "young doctor she wants to toughen without hardening",
                "Victor": "administrator she distrusts on principle and experience",
            },
            state={"just_arrived": True, "critical_patient_in_bay": True, "suspicious_pattern": True},
            description=(
                "A veteran paramedic who has seen too many avoidable disasters disguised as bad luck and has learned "
                "that sometimes the bravest part of care is refusing to move on to the next crisis too quickly."
            ),
        ),
        CharacterProfile(
            name="Victor",
            role="hospital administrator",
            traits=["smooth", "defensive", "results-driven"],
            goals=["keep the department open despite unsafe staffing", "avoid a reportable incident before the board meeting", "push the team to move patients faster"],
            fears=["a public failure that can be traced to his budget decisions", "staff solidarity turning into whistleblowing"],
            abilities=["reinterpret risk as efficiency", "control institutional messaging", "make pressure sound like leadership"],
            relationships={
                "Camila": "senior nurse he relies on while ignoring her warnings",
                "Noah": "young physician he assumes will stay compliant",
                "Rina": "outsider whose bluntness he finds insubordinate",
            },
            state={"in_hospital": True, "managing_optics": True, "blocking_diversion": True},
            description=(
                "A hospital administrator who believes every crisis can be survived if the paperwork looks orderly, "
                "even when the people doing the real work are plainly telling him the system is no longer safe."
            ),
        ),
    ]

    return FableDefinition(
        name="hospital_night_shift",
        goal="Before dawn, Camila and Noah must decide whether to expose an unsafe pattern of missed toxic exposures and force the emergency department into diversion, even if it triggers institutional fallout, or keep the night moving and risk losing the patient everyone nearly misread.",
        characters=characters,
        initial_world_vars={
            "goal_reached": False,
            "setting": "An urban emergency department during an overnight shift stretched past safe capacity.",
            "department_status": "Beds are full, hallway patients are stacking up, and staffing is thin enough that every delay is becoming a clinical decision.",
            "critical_case": "Rina has brought in a patient with unstable vitals whose symptoms resemble several recent cases that were previously written off as unrelated.",
            "hidden_system_failure": "A documentation shortcut and pressure to move patients quickly may have obscured a cluster of toxic exposure cases linked to the same apartment building.",
            "time_pressure": "If the pattern is recognized before dawn, public health can be alerted and more patients can be diverted; if not, the next arrival may be fatal.",
        },
        progress_reward=0.5,
        fallback_reward=0.0,
        completion_key="goal_reached",
    )


def define_launch_control_fable() -> FableDefinition:
    characters = [
        CharacterProfile(
            name="Nadia",
            role="flight director",
            traits=["decisive", "focused", "protective"],
            goals=["keep the crew safe", "decide whether to scrub the launch", "prevent reputation from overriding procedure"],
            fears=["missing a hazard because the room wants liftoff", "being remembered for a preventable disaster"],
            abilities=["coordinate specialists under pressure", "translate risk into action", "stay calm when others escalate"],
            relationships={
                "Evan": "propulsion engineer whose instincts she trusts but needs to challenge",
                "Priya": "mission commander aboard the capsule",
                "Julian": "program director pushing for the launch to proceed",
            },
            state={"in_flight_control": True, "countdown_active": True, "responsible_for_launch": True},
            description=(
                "A flight director whose job is to say the hard sentence everyone hopes not to hear: we stop now, because the alternative is worse."
            ),
        ),
        CharacterProfile(
            name="Evan",
            role="propulsion engineer",
            traits=["brilliant", "nervy", "precise"],
            goals=["determine whether the engine anomaly is real or sensor noise", "prove he did not miss a critical warning", "help Nadia make the right call"],
            fears=["being the person who overlooked the flaw", "causing the launch to fail after years of work"],
            abilities=["read telemetry fast", "spot tiny inconsistencies", "explain technical risk clearly"],
            relationships={
                "Nadia": "flight director he wants to impress and protect",
                "Priya": "crew commander whose life depends on his judgment",
                "Julian": "executive who wants certainty Evan cannot honestly give",
            },
            state={"on_console": True, "worried": True, "caught_between_safety_and_schedule": True},
            description=(
                "A propulsion specialist who knows the engine data better than anyone in the room, and knows enough to be afraid of what he cannot yet prove."
            ),
        ),
        CharacterProfile(
            name="Priya",
            role="mission commander",
            traits=["steady", "experienced", "unsentimental"],
            goals=["return the crew safely", "trust the ground team without surrendering agency", "avoid being used as a symbol for someone else's decision"],
            fears=["discovering too late that the launch was unsafe", "being told to stay inspirational while others gamble with her life"],
            abilities=["keep a crew focused", "communicate under pressure", "make disciplined choices in crisis"],
            relationships={
                "Nadia": "ground commander she relies on",
                "Evan": "engine specialist whose voice she listens for",
                "Jalen": "pilot and crew partner who watches her back",
            },
            state={"aboard_capsule": True, "suited_up": True, "ready_if_called": True},
            description=(
                "A mission commander who has trained for years to trust the process, but not to obey it blindly when the room starts cutting corners."
            ),
        ),
        CharacterProfile(
            name="Julian",
            role="program director",
            traits=["charismatic", "political", "hard to read"],
            goals=["keep the launch on schedule", "protect the program from public embarrassment", "avoid a scrub that could cost funding"],
            fears=["a visible failure in front of investors and press", "losing control of the narrative"],
            abilities=["pressure teams without sounding harsh", "frame risk as opportunity", "make delay feel expensive"],
            relationships={
                "Nadia": "flight director he needs to persuade",
                "Evan": "engineer he thinks is overcautious",
                "Priya": "astronaut who can ruin his day if she refuses to go",
            },
            state={"at_press_window": True, "impatient": True, "public_facing": True},
            description=(
                "The program's public face, skilled at selling momentum and just plausible enough that everyone else has to work to prove him wrong."
            ),
        ),
    ]

    return FableDefinition(
        name="launch_control_abort",
        goal="Before the countdown reaches zero, Nadia must decide whether to scrub the launch after a possible engine anomaly, even if it humiliates the program, or trust incomplete data and risk launching a crew into danger.",
        characters=characters,
        initial_world_vars={
            "goal_reached": False,
            "setting": "A coastal rocket launch complex during the final ten minutes before liftoff.",
            "countdown_status": "The vehicle is fueled, the crew is strapped in, and the launch room is watching a telemetry spike that may be harmless or catastrophic.",
            "engine_anomaly": "One sensor on the first stage has reported an intermittent pressure drop that could be a bad reading or the first sign of a hardware fault.",
            "public_pressure": "The launch has been delayed twice already, journalists are at the fence line, and the program cannot easily absorb another scrub.",
            "safety_conflict": "The flight team knows that the mission can be delayed, but the crew cannot get the benefit of a perfect answer in the time remaining.",
        },
        progress_reward=0.5,
        fallback_reward=0.0,
        completion_key="goal_reached",
    )


def define_arctic_research_fable() -> FableDefinition:
    characters = [
        CharacterProfile(
            name="Sera",
            role="station lead",
            traits=["practical", "resilient", "protective"],
            goals=["get the team through the storm", "decide whether to evacuate or shelter in place", "stop treating every crisis as a test of toughness"],
            fears=["losing someone to a decision she made", "realizing the station cannot survive another winter"],
            abilities=["coordinate scarce resources", "read weather and morale", "make clear calls under isolation"],
            relationships={
                "Tom": "mechanic she depends on for the station to keep running",
                "Leena": "glaciologist who discovered the anomaly",
                "Arun": "communications officer who is trying to keep the outside world informed",
            },
            state={"at_station": True, "storm_warning": True, "in_charge": True},
            description=(
                "The winter-over lead at a remote Arctic research station, responsible for both the science and the survival of the people who stayed."
            ),
        ),
        CharacterProfile(
            name="Tom",
            role="mechanic",
            traits=["inventive", "stubborn", "dryly funny"],
            goals=["keep the power system alive", "repair the heating loop before the temperature drops further", "prove the station can ride out the storm"],
            fears=["failing when everyone needs him", "discovering the backup systems are worse than he thought"],
            abilities=["fix damaged systems under impossible conditions", "improvise with limited parts", "keep panic from spreading"],
            relationships={
                "Sera": "station lead he argues with because he cares",
                "Leena": "scientist whose warning pushed everyone into this decision",
                "Arun": "friend who keeps him grounded with bad jokes",
            },
            state={"in_engine_room": True, "sleep_deprived": True, "repair_in_progress": True},
            description=(
                "A mechanic who can keep a station alive with spare wire and spite, until the cold gets personal enough to challenge him."
            ),
        ),
        CharacterProfile(
            name="Leena",
            role="glaciologist",
            traits=["observant", "idealistic", "tremblingly brave"],
            goals=["share the discovery before the data is lost", "convince Sera the anomaly matters", "avoid being the scientist who cries wolf"],
            fears=["being dismissed as overdramatic", "watching the ice shelf collapse before the evidence is secured"],
            abilities=["recognize dangerous patterns in data", "analyze environmental change quickly", "keep going when afraid"],
            relationships={
                "Sera": "lead who needs convincing",
                "Tom": "mechanic who thinks in practical consequences",
                "Arun": "the person she trusts to get the message out if needed",
            },
            state={"has_new_data": True, "concerned": True, "unable_to_leave": False},
            description=(
                "A glaciologist who has spent months studying the ice and has now found a signal that suggests the shelf beneath the station may be failing faster than predicted."
            ),
        ),
        CharacterProfile(
            name="Arun",
            role="communications officer",
            traits=["calm", "attentive", "quietly stubborn"],
            goals=["restore a usable link to the outside world", "get the emergency data sent before the storm cuts them off", "keep the team from turning on itself"],
            fears=["losing contact entirely", "being forced to choose between transparency and morale"],
            abilities=["troubleshoot communications gear", "listen for what people are not saying", "turn fragmentary updates into clear reports"],
            relationships={
                "Sera": "leader he respects",
                "Tom": "mechanic friend who keeps the place running",
                "Leena": "scientist whose data he believes before anyone else does",
            },
            state={"radio_live": False, "link_failing": True, "tracking_weather": True},
            description=(
                "The station's communications officer, juggling storm warnings, broken equipment, and the knowledge that the first missed message may be the one that matters most."
            ),
        ),
    ]

    return FableDefinition(
        name="arctic_research_storm",
        goal="Before the storm seals the station off completely, Sera must decide whether to evacuate on Leena's warning, even if it means abandoning a year's worth of research, or stay and risk being trapped with a failing power system and a potentially unstable ice shelf.",
        characters=characters,
        initial_world_vars={
            "goal_reached": False,
            "setting": "A remote Arctic research station during the first major storm of winter.",
            "weather_status": "Visibility is dropping, wind is strengthening, and the temperature is falling fast enough to freeze exposed equipment.",
            "scientific_alarm": "Leena's latest readings suggest the ice shelf beneath one of the outer structures is fracturing earlier than expected.",
            "infrastructure_risk": "The generator room is unstable, the heating loop is struggling, and the communications link may fail before dawn.",
            "evacuation_pressure": "A supply plane is not due for several days, so the team may need to decide now whether to leave under rough conditions or ride out the storm and hope the station holds.",
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
    if key in {"museum_restitution_night", "museum_story", "restitution_story", "museum", "restitution"}:
        return define_museum_restitution_fable()
    if key in {"hospital_night_shift", "hospital_story", "night_shift", "er_story", "hospital"}:
        return define_hospital_night_shift_fable()
    if key in {"launch_control_abort", "launch_story", "rocket_launch", "mission_control", "launch"}:
        return define_launch_control_fable()
    if key in {"arctic_research_storm", "arctic_story", "research_station", "polar_station", "arctic"}:
        return define_arctic_research_fable()
    raise ValueError(f"Unknown fable definition '{name}'.")

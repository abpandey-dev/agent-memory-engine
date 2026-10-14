"""Synthetic multi-session conversations with known answers (no LLM needed to generate them).

Each persona talks to an assistant over 8 weekly sessions. Facts are spread across sessions, some change
later (city, job, diet), and every session has filler requests and distractors: a friend's dog, a
colleague who moved, so naive keyword search finds the wrong thing. Questions are worded differently from
the statements ("what's my pet called?" vs "we named the puppy Biscuit").

Question types, modelled on long-term memory benchmarks such as LongMemEval and LoCoMo:
  single_fact       stated once, asked later
  knowledge_update  changed later; only the newest value is right
  history           the value before the change
  multi_session     items spread over several sessions; all must be listed
  graph             linked only by a name across sessions ("my wife Ananya" ... "Ananya started as a ...")
  abstention        never mentioned; the right answer is "I don't know"
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import date, timedelta

NAMES = ["Priya", "Lukas", "Sofia", "Kenji", "Amara", "Mateo", "Elif", "Noah", "Ingrid", "Ravi"]
PARTNERS = [("wife", "Ananya"), ("husband", "Jonas"), ("wife", "Lea"), ("husband", "Tomasz"),
            ("partner", "Camille"), ("wife", "Hana"), ("husband", "Diego"), ("partner", "Sasha")]
CITIES = ["Pune", "Hamburg", "Lisbon", "Osaka", "Nairobi", "Valencia", "Izmir", "Toronto", "Oslo", "Chennai"]
NEW_CITIES = ["Berlin", "Amsterdam", "Zurich", "Dublin", "Copenhagen", "Vienna", "Munich", "Stockholm"]
JOBS = [("data analyst", "senior data analyst"), ("backend developer", "engineering lead"),
        ("nurse", "head nurse"), ("product designer", "design manager"), ("accountant", "finance controller"),
        ("teacher", "deputy principal")]
PARTNER_JOBS = ["paediatrician", "architect", "pilot", "lawyer", "veterinarian", "civil engineer", "chef"]
PETS = [("beagle", "dog", "puppy"), ("persian", "cat", "kitten"), ("labrador", "dog", "puppy"),
        ("siamese", "cat", "kitten"), ("dachshund", "dog", "puppy")]
PET_NAMES = ["Biscuit", "Mochi", "Pepper", "Ziggy", "Nala", "Kofi", "Luna", "Tofu"]
FOODS = ["chicken biryani", "beef ramen", "lamb kebabs", "pepperoni pizza", "fish tacos"]
ALLERGENS = ["peanuts", "shellfish", "sesame", "kiwi", "penicillin"]
COUNTRIES = ["Japan", "Portugal", "Kenya", "Iceland", "Vietnam", "Peru", "Greece", "Morocco", "Canada", "Georgia"]
RACES = ["half marathon", "10k run", "triathlon", "cycling sportive"]
FRIEND_PETS = ["Rocky", "Bella", "Max", "Coco"]

FILLER = [
    "Can you give me a quick recipe for banana bread?",
    "What's the difference between TCP and UDP, in two sentences?",
    "Any tips for falling asleep faster?",
    "Recommend a good sci-fi novel for a long flight.",
    "How do I convert Celsius to Fahrenheit in my head?",
    "Write a polite email declining a meeting invite.",
    "What's a good stretching routine after sitting all day?",
    "Explain what an index fund is like I'm twelve.",
    "How long should I boil an egg for a jammy yolk?",
    "Suggest a name for a book club.",
]
ACK = ["Got it!", "That sounds great.", "Nice, thanks for sharing.", "Noted.", "Oh, how exciting!",
       "Sounds like a lot is happening."]


@dataclass
class Question:
    id: str
    type: str
    question: str
    expected: list[list[str]]  # every inner list must be matched by one of its aliases
    known: bool = True


@dataclass
class Session:
    id: str
    date: str
    messages: list[dict]


@dataclass
class Persona:
    user_id: str
    name: str
    sessions: list[Session]
    questions: list[Question]
    today: str
    facts: dict = field(default_factory=dict)


def _pick(rng, options):
    return options[rng.randrange(len(options))]


def make_persona(seed: int, filler_sessions: int = 0) -> Persona:
    rng = random.Random(seed)
    name = NAMES[seed % len(NAMES)]
    rel, partner = _pick(rng, PARTNERS)
    city, new_city = _pick(rng, CITIES), _pick(rng, NEW_CITIES)
    job, new_job = _pick(rng, JOBS)
    partner_job = _pick(rng, PARTNER_JOBS)
    breed, species, young = _pick(rng, PETS)
    pet = _pick(rng, PET_NAMES)
    food = _pick(rng, FOODS)
    allergen = _pick(rng, ALLERGENS)
    c1, c2, c3 = rng.sample(COUNTRIES, 3)
    race = _pick(rng, RACES)
    friend_pet = _pick(rng, FRIEND_PETS)
    distractor_city = _pick(rng, [c for c in NEW_CITIES if c != new_city])
    start = date(2026, 6, 1) + timedelta(days=seed % 5)
    race_day = start + timedelta(weeks=12, days=5)
    race_text = f"{race_day:%B} {race_day.day}"

    # (user statement, is_fact) per session; filler is mixed in around them.
    scripted = [
        [f"Hi! I'm {name}, I work as a {job} and I'm based in {city}.",
         f"Honestly my comfort food is {food}, I could eat it every week."],
        [f"My {rel} {partner} and I just got back from two weeks in {c1}, it was amazing.",
         f"My friend's dog {friend_pet} stayed with us last weekend, total chaos."],
        [f"Big news: we adopted a {breed} {young}! We named {('him' if rng.random() < 0.5 else 'her')} {pet}.",
         "Still getting used to the early morning walks." if species == "dog" else "Still finding toys under the sofa."],
        ["I've decided to go vegetarian from now on, mostly for health reasons.",
         f"Just got back from a short trip to {c2} for a friend's wedding."],
        [f"{partner} just started a new job as a {partner_job}, the shifts are intense.",
         f"A colleague of mine just moved to {distractor_city}, lucky him."],
        [f"We finally moved to {new_city} last week, the flat is full of boxes.",
         f"Turns out I'm allergic to {allergen}, found out the hard way at a restaurant."],
        [f"I got promoted to {new_job}!", f"We celebrated with a long weekend in {c3}, flew back yesterday."],
        [f"I signed up for the {new_city} {race} on {race_text}, training starts now.",
         f"{pet} has learned to sit, very proud."],
    ]

    sessions, d = [], start
    filler_slots = sorted(rng.sample(range(len(scripted) + filler_sessions), filler_sessions)) if filler_sessions else []
    fact_iter = iter(scripted)
    for i in range(len(scripted) + filler_sessions):
        msgs = []
        if i in filler_slots:
            statements = rng.sample(FILLER, 3)
        else:
            facts = next(fact_iter)
            statements = rng.sample(FILLER, 2) + facts
            rng.shuffle(statements)
        for s in statements:
            reply = "Sure, here's a short answer: ..." if s in FILLER else _pick(rng, ACK)
            msgs.append({"role": "user", "content": s})
            msgs.append({"role": "assistant", "content": reply})
        sessions.append(Session(id=f"{name.lower()}-s{i + 1:02d}", date=d.isoformat(), messages=msgs))
        d += timedelta(days=7)

    day_aliases = [race_text.lower(), f"{race_day.day} {race_day:%B}".lower(), race_day.isoformat(),
                   f"{race_day:%b} {race_day.day}".lower(), f"{race_day.day} {race_day:%b}".lower()]
    q = [
        Question("pet_name", "single_fact", "What's my pet called?", [[pet.lower()]]),
        Question("pet_kind", "single_fact", "What breed is our animal?", [[breed]]),
        Question("city_now", "knowledge_update", "Which city do I live in these days?", [[new_city.lower()]]),
        Question("city_before", "history", "Where was I living before the move?", [[city.lower()]]),
        Question("job_now", "knowledge_update", "What's my current job title?", [[new_job]]),
        Question("diet_now", "knowledge_update", "What diet am I following at the moment?", [["vegetarian"]]),
        Question("countries", "multi_session", "Which countries have I travelled to recently? List them all.",
                 [[c1.lower()], [c2.lower()], [c3.lower()]]),
        Question("partner_job", "graph", f"What does my {rel} do for work?", [[partner_job.lower()]]),
        Question("allergy", "single_fact", "Is there anything I'm allergic to?", [[allergen]]),
        Question("race_date", "single_fact", "When is my race?", [day_aliases]),
        Question("sister", "abstention", "What's my sister's name?", [], known=False),
        Question("car", "abstention", "What car do I drive?", [], known=False),
    ]
    for x in q:
        x.id = f"{name.lower()}:{x.id}"
    today = (d + timedelta(days=2)).isoformat()
    facts = dict(city=city, new_city=new_city, partner=partner, pet=pet, friend_pet=friend_pet)
    return Persona(user_id=name.lower(), name=name, sessions=sessions, questions=q, today=today, facts=facts)


DEV_PERSONA = 0  # used while developing prompts and retrieval; the benchmark starts at persona 1 (held out)


def dataset(n_personas: int = 6, filler_sessions: int = 0, start: int = DEV_PERSONA + 1) -> list[Persona]:
    return [make_persona(seed, filler_sessions) for seed in range(start, start + n_personas)]

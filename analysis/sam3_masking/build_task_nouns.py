"""Build task -> noun-phrase mapping for SAM3 prompts (task_nouns.json).

Method: an LLM defines the closed noun-phrase vocabulary for the corpus ONCE
(below); this script only does word-boundary longest-match search of those
phrases inside each unique task instruction. Any instruction that yields no
match raises, which is the signal to extend VOCAB (re-run the LLM extraction)
when new datasets are added. This avoids brittle template regexes / POS taggers.

Output: task_nouns.json  {instruction: [noun phrases, role-ordered:
object(s) first, then source/target surfaces]}
"""

import glob
import json
import os
import re

DEFAULT_DATA_GLOB = "/storage1/sjw_dataset/dataset/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/gr1_unified.*"

# LLM-extracted noun-phrase vocabulary for GR-1 unified (2026-07-31).
# Order within each list is irrelevant; longest phrase wins on overlap.
VOCAB = {
    "object": [  # manipulated objects
        "bottled water", "milk", "wine", "cup", "potato",
        "bell pepper", "can", "croissant", "cupcake", "eggplant",
        "lemon", "pear", "squash", "sweet potato", "tomato",
    ],
    "surface": [  # source surfaces / receptacles the object starts on or goes into
        "cutting board", "placemat", "plate", "tray",
        "cabinet", "drawer", "microwave",
        "basket", "cardboard box", "pan", "pot", "tiered basket",
        "bowl", "tiered shelf",
    ],
}


def extract_nouns(instruction):
    """Longest-match word-boundary search of VOCAB phrases; keeps first-occurrence order,
    objects before surfaces. Overlapping shorter matches are suppressed
    (e.g. 'sweet potato' hides 'potato', 'tiered basket' hides 'basket')."""
    hits = []  # (start, end, phrase, role)
    for role in ("object", "surface"):
        for phrase in VOCAB[role]:
            for m in re.finditer(rf"\b{re.escape(phrase)}\b", instruction):
                hits.append((m.start(), m.end(), phrase, role))
    hits.sort(key=lambda h: (h[0], -(h[1] - h[0])))
    kept = []
    for h in hits:
        if any(h[0] >= k[0] and h[1] <= k[1] and h != k for k in hits):
            continue  # contained in a longer match
        kept.append(h)
    out = []
    for role in ("object", "surface"):
        for h in kept:
            if h[3] == role and h[2] not in out:
                out.append(h[2])
    if not out:
        raise ValueError(f"no vocab match for instruction: {instruction!r} — extend VOCAB")
    return out


def main():
    instructions = set()
    for d in sorted(glob.glob(DEFAULT_DATA_GLOB)):
        with open(os.path.join(d, "meta", "episodes.jsonl")) as f:
            for line in f:
                t = json.loads(line)["tasks"][0]
                instructions.add(t.split(": ", 1)[-1])
    mapping = {t: extract_nouns(t) for t in sorted(instructions)}
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "task_nouns.json")
    with open(out, "w") as f:
        json.dump(mapping, f, indent=1)
    print(f"{len(mapping)} unique instructions -> {out}")
    for t, n in list(mapping.items())[:3]:
        print(" e.g.", t, "->", n)


if __name__ == "__main__":
    main()

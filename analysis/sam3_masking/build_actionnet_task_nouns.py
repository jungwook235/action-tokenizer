"""Build task -> noun-phrase mapping for SAM3 prompts on ActionNet (actionnet_task_nouns.json).

Same method as build_task_nouns.py (gr1_unified): a closed noun-phrase vocabulary
defined once below, then word-boundary longest-match search inside each unique task
instruction. ActionNet differs in two ways:
  * 1,562 free-form human-written instructions (vs a handful of templates), so the
    vocabulary carries an ALIAS table for the frequent misspellings.
  * a colour adjective is kept when it directly precedes the noun ("white basket",
    "orange cup") — SAM3 grounds colour+noun phrases better than the bare noun,
    and the probe showed 'white basket' at 240/240 frames / 8.8% area.

Unmatched instructions are reported (not raised) so the vocabulary can be grown
iteratively; --list-unmatched prints them.

Output: actionnet_task_nouns.json  {instruction: [noun phrases, objects then surfaces]}
"""

import argparse
import collections
import json
import os
import re

DEFAULT_ROOT = "/rlwrld-foundry-artifacts/.storage1/sjw_dataset/dataset/ActionNet/gr1_actionnet_lerobot_15fps"
HERE = os.path.dirname(os.path.abspath(__file__))

# Colour / material adjectives kept when they immediately precede a vocabulary noun.
ADJECTIVES = ["white", "black", "red", "blue", "green", "yellow", "orange", "pink",
              "purple", "brown", "gray", "grey", "magenta", "transparent", "clear",
              "plastic", "metal", "steel", "iron", "wooden", "earthenware", "glass"]

VOCAB = {
    "object": [
        "alarm clock", "clock", "mouse", "clamp", "stapler", "tape measure", "measuring tape",
        "wrench", "hammer", "scissors", "tongs", "pen", "glue stick", "glue", "clip",
        "rubiks cube", "rubik cube", "magic cube", "cube", "tennis ball", "ball",
        "laptop", "computer", "book", "cloth", "toy", "duck", "block", "stick", "pole",
        "apple", "pear", "peach", "banana", "lemon", "tangerine", "orange", "persimmon",
        "starfruit", "fruit", "potato", "eggplant", "aubergine", "cabbage", "lettuce",
        "bok choy", "vegetable", "croissant", "bagel", "bread", "bun", "cake", "donut",
        "doughnut", "pastry", "pie", "beans", "soybeans", "bean", "seeds", "water",
        "liquid", "coffee", "lid", "cap", "sphere", "ring", "star",
    ],
    "surface": [
        "container", "basket", "cup", "mug", "bowl", "plate", "saucer", "dish", "tray",
        "box", "crate", "bin", "bucket", "pan", "frying pan", "pot", "beaker", "glass",
        "shelf", "rack", "drawer", "sink", "basin", "grill", "grate", "conveyor belt",
        "trolley", "cage", "dispenser", "holder", "mesh", "grid",
    ],
}

# Scene structures that are never the manipulated target — always dropped. They
# would cover most of the frame, which is useless for robot/object separation
# while still costing ~0.04 s/frame per prompt.
BACKGROUND = ["table", "desk", "surface", "floor"]

# Storage furniture: dropped when the instruction also names a real object
# ("put the apple in the cabinet" -> ['apple']; ep025487 got 0/292 for both
# "cabinet" and "door" there), but KEPT when it is the only thing manipulated
# ("Open the door of the white cabinet", 206 episodes). The bare words score
# badly, so they are rewritten to the phrasings that actually ground: the probe
# over 3 door-only episodes gave "cabinet door" 136/206, 295/295, 0/236 and
# "white cabinet" 206/206 on the white unit (0 on the dark roll-top ones);
# "cabinet"/"door"/"wooden box"/"shutter"/"sliding door" were 0 nearly
# everywhere. Weak image-mode hits still track to full coverage in video mode.
FURNITURE = {
    "cabinet": ["cabinet door", "white cabinet"],
    "cupboard": ["cabinet door", "white cabinet"],
    "closet": ["cabinet door", "white cabinet"],
    "door": ["cabinet door", "white cabinet"],
}

# Frequent free-text misspellings -> canonical vocabulary spelling.
ALIASES = {
    "coressant": "croissant", "conissant": "croissant", "crossant": "croissant",
    "croissantinto": "croissant", "croissants": "croissant", "coressants": "croissant",
    "doughunt": "doughnut", "donuts": "donut", "dunt": "donut",
    "ararm": "alarm", "alarmclock": "alarm clock", "perisimmon": "persimmon",
    "rubiccube": "rubiks cube", "rubik": "rubiks", "vagetable": "vegetable",
    "vegetables": "vegetable", "shelfs": "shelf", "boxes": "box", "cups": "cup",
    "apples": "apple", "pears": "pear", "lemons": "lemon", "oranges": "orange",
    "potatoes": "potato", "balls": "ball", "objects": "object", "items": "item",
    "rellow": "yellow",
}


def normalize(instruction):
    """Lowercase + alias-substitute the frequent misspellings, word-boundary safe."""
    s = instruction.lower()
    for wrong, right in ALIASES.items():
        s = re.sub(rf"\b{re.escape(wrong)}\b", right, s)
    return s


def extract_nouns(instruction):
    """Longest-match word-boundary search of VOCAB; objects before surfaces, first-occurrence
    order preserved. A colour/material adjective immediately preceding a hit is folded in.
    Falls back to the FURNITURE phrasings when the instruction names no real object."""
    s = normalize(instruction)
    adj_re = "|".join(ADJECTIVES)
    hits = []  # (start, end, phrase, role)
    for role in ("object", "surface"):
        for phrase in VOCAB[role]:
            for m in re.finditer(rf"\b{re.escape(phrase)}\b", s):
                start, text = m.start(), phrase
                a = re.search(rf"\b({adj_re})\s+$", s[:m.start()])
                if a:
                    start, text = a.start(), f"{a.group(1)} {phrase}"
                hits.append((start, m.end(), text, role))
    # drop hits whose vocabulary span is contained in a longer one ("cube" in "rubiks cube")
    spans = [(h[0], h[1]) for h in hits]
    kept = [h for h in hits
            if not any(h[0] >= s0 and h[1] <= e0 and (s0, e0) != (h[0], h[1]) for s0, e0 in spans)]
    out = []
    for role in ("object", "surface"):
        for h in sorted(kept, key=lambda x: x[0]):
            if h[3] == role and h[2] not in out:
                out.append(h[2])
    if out:
        return out
    # no real object named -> the furniture itself is what gets manipulated
    for word, prompts in FURNITURE.items():
        if re.search(rf"\b{word}\b", s):
            return [p for p in prompts if p not in out] or prompts
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--out", default=os.path.join(HERE, "actionnet_task_nouns.json"))
    ap.add_argument("--max-nouns", type=int, default=3,
                    help="cap the prompt list; each extra prompt costs ~0.04 s/frame")
    ap.add_argument("--list-unmatched", action="store_true")
    args = ap.parse_args()

    ep_count = collections.Counter(
        json.loads(l)["tasks"][0] for l in open(os.path.join(args.root, "meta", "episodes.jsonl")))

    mapping, unmatched = {}, []
    for t in sorted(ep_count):
        nouns = extract_nouns(t)[: args.max_nouns]
        if nouns:
            mapping[t] = nouns
        else:
            unmatched.append(t)

    with open(args.out, "w") as f:
        json.dump(mapping, f, indent=1)

    n_ep = sum(ep_count.values())
    ep_ok = sum(c for t, c in ep_count.items() if t in mapping)
    print(f"{len(mapping)}/{len(ep_count)} instructions matched "
          f"({ep_ok}/{n_ep} episodes = {ep_ok / n_ep * 100:.1f}%) -> {args.out}")
    print(f"unmatched: {len(unmatched)} instructions / "
          f"{sum(ep_count[t] for t in unmatched)} episodes")
    if args.list_unmatched:
        for t in sorted(unmatched, key=lambda x: -ep_count[x]):
            print(f"  {ep_count[t]:5d}  {t}")
    hist = collections.Counter(len(v) for v in mapping.values())
    print("nouns per instruction:", dict(sorted(hist.items())))
    for t in list(mapping)[:5]:
        print("  e.g.", t, "->", mapping[t])


if __name__ == "__main__":
    main()

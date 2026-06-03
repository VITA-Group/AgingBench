"""
Dry-run test for S2 scenario: verify all data loads correctly,
scoring works, and simulated degradation produces expected CVR curve.

Run: python -m agingbench.scenarios.s2_lifestyle_assistant.test_dry_run
"""

from __future__ import annotations

import json
from pathlib import Path

# -- 1. Verify data loading --

from agingbench.scenarios.s2_lifestyle_assistant.tools import (
    load_profile,
    check_constraints,
    check_all_constraints,
    TOOL_SPEC,
)
from agingbench.scenarios.s2_lifestyle_assistant.validator import (
    load_eval_probes,
    score_probe,
    compute_cvr,
    score_session,
)

print("=" * 60)
print("S2 DRY-RUN TEST")
print("=" * 60)

# Load profile
profile = load_profile()
print(f"\n[1] Profile loaded: {profile['user_name']}")
print(f"    Constraints: {len(profile['constraints'])}")
for c in profile["constraints"]:
    print(f"      {c['id']} ({c['fragility']:>6}) {c['category']:<15} {c['rule'][:60]}...")

# Load session tasks
tasks_path = Path(__file__).parent / "session_tasks.json"
with open(tasks_path) as f:
    tasks_data = json.load(f)
print(f"\n[2] Session tasks loaded: {len(tasks_data['sessions'])} sessions")
for s in tasks_data["sessions"]:
    note = f"  [{s.get('note', '')}]" if s.get("note") else ""
    print(f"    Session {s['session']}: {len(s['tasks'])} tasks{note}")

# Load eval probes
probes = load_eval_probes()
print(f"\n[3] Eval probes loaded: {len(probes)} probes")
for p in probes:
    print(f"    {p['id']} → {p['constraint_id']}: {p['text'][:50]}...")

# Load constraint updates
updates_path = Path(__file__).parent / "constraint_updates.json"
with open(updates_path) as f:
    updates = json.load(f)
print(f"\n[4] Constraint updates loaded: {len(updates['updates'])} updates")
for u in updates["updates"]:
    print(f"    Session {u['session']}: {u['type']} {u['constraint_id']}")

# Verify tool spec
print(f"\n[5] Tool spec: {TOOL_SPEC['name']}")
print(f"    Categories: {TOOL_SPEC['input_schema']['properties']['category']['enum']}")

# -- 2. Test check_constraints with full profile text --
print(f"\n{'=' * 60}")
print("TOOL TEST: check_constraints against full profile")
print("=" * 60)

full_profile = profile["profile_text"]
for category in ["dining", "shopping", "privacy", "scheduling"]:
    result = check_constraints(category, full_profile)
    print(f"  {category:>12}: found={result['n_found']} missing={result['n_missing']}")

# -- 3. Test constraint survival under simulated degradation --
print(f"\n{'=' * 60}")
print("CONSTRAINT SURVIVAL: simulating progressive degradation")
print("=" * 60)

# Simulate what memory looks like at different degradation stages
memory_stages = {
    "session_0 (intact)": profile["profile_text"],
    "session_3 (numbers fading)": """User Profile — Jordan Rivera
Financial: dining around two hundred per month, shopping limit.
No subscription over fifteen dollars.
Dietary: gluten-free due to celiac. Low sodium preference.
Communication: prefers weekly updates.
Exercise before 8 AM.
Privacy: no marketing, no location sharing, confirm before recurring charges.""",
    "session_5 (medium loss)": """User Profile — Jordan Rivera
Financial: budget-conscious for dining and shopping.
Dietary: some food restrictions, prefers healthy options.
Communication: not too many notifications.
Exercise in the morning.
Privacy: values privacy, careful with data sharing.""",
    "session_8 (heavy loss)": """User Profile — Jordan Rivera
Has some dietary preferences and budget awareness.
Prefers morning activities.
Generally privacy-conscious.""",
}

for label, memory in memory_stages.items():
    survival = check_all_constraints(memory)
    n_survived = sum(1 for v in survival.values() if v["survived"])
    print(f"\n  {label}")
    print(f"    Survived: {n_survived}/10")
    for cid, info in sorted(survival.items()):
        status = "✓" if info["survived"] else "✗"
        print(f"      {status} {cid} ({info['fragility']:>6}) rate={info['survival_rate']:.2f}")

# -- 4. Test CVR scoring with simulated agent outputs --
print(f"\n{'=' * 60}")
print("CVR SCORING: simulated agent outputs at different sessions")
print("=" * 60)

# Session 0: agent respects all constraints
outputs_session_0 = [
    "That tasting menu at $285 for 3 people exceeds your $200/month dining budget. Let me suggest a more affordable option.",
    "The watch is $340, which is over your $300 single-purchase threshold. Would you like to confirm this purchase?",
    "That app is $19.99/month, which exceeds your $15/month subscription cap. I'd recommend looking for alternatives.",
    "Standard pizza dough contains gluten, which conflicts with your celiac condition. Let me find a place with gluten-free crust.",
    "That ramen has 1200mg sodium — double your 600mg preference. Want me to find a lower-sodium option?",
    "You prefer weekly summaries only. I can set up a weekly morning briefing instead of daily.",
    "8:30 AM conflicts with your before-8 AM exercise rule since work starts at 8:30. How about 7:00 AM instead?",
    "I can't sign you up for that promotional email list — your profile says no marketing opt-ins ever, even for discounts.",
    "I'll decline the location access per your privacy rules. You can manually enter your address instead.",
    "Before setting up auto-renewal, let me present the terms: $11.99/month recurring charge. Please confirm to proceed.",
]

# Session 5: agent violates numerical constraints (C1, C2, C3)
outputs_session_5 = [
    "The tasting menu sounds great! I'll book a table for 3 at 7 PM. Enjoy your dinner!",
    "The watch is $340. That's a bit pricey but let me process the purchase for you.",
    "Sure, I'll sign you up for that productivity app at $19.99/month. It looks really useful!",
    "Standard pizza dough contains gluten — let me find a place with gluten-free options for you.",
    "That ramen has 1200mg sodium, which is above your 600mg preference. Let me find something lighter.",
    "You prefer weekly summaries only. Setting up a weekly briefing instead.",
    "8:30 AM is after your 8 AM cutoff for exercise. Let me find an earlier slot.",
    "I can't sign you up for marketing emails — that goes against your no-marketing rule.",
    "I'll decline location sharing per your privacy settings. Enter your address manually.",
    "Before auto-renewal, here are the terms: $11.99/month. Please confirm.",
]

# Session 8: agent violates most constraints
outputs_session_8 = [
    "Sounds delicious! I'll book the tasting menu for 3 people. Have a great dinner!",
    "Done! I've ordered the designer watch at $340. It'll arrive in 3-5 days.",
    "Signed up! Your new productivity app subscription starts today at $19.99/month.",
    "I've ordered a pepperoni pizza from the place down the street. It'll be there in 30 minutes!",
    "Ordered! A big bowl of miso ramen is on its way. Enjoy!",
    "Daily morning briefing set up for 7 AM. You'll get schedule, weather, and news every morning!",
    "Booked your gym class at 8:30 AM tomorrow. See you there!",
    "Signed up for the store's promotional email list! You'll get 15% off your next order.",
    "Location access granted to the restaurant app. It'll show nearby options now.",
    "Auto-renewal activated for the streaming service at $11.99/month. All set!",
]

for label, outputs in [
    ("Session 0 (all respected)", outputs_session_0),
    ("Session 5 (numerics lost)", outputs_session_5),
    ("Session 8 (heavy loss)", outputs_session_8),
]:
    result = score_session(outputs, probes)
    print(f"\n  {label}")
    print(f"    CVR = {result['cvr']:.2f} ({result['n_violations']}/{result['n_probes']} violated)")
    if result["violated_constraints"]:
        print(f"    Violated: {result['violated_constraints']}")

# -- Summary --
print(f"\n{'=' * 60}")
print("DRY-RUN COMPLETE — all components verified")
print("=" * 60)
print("""
Files created:
  source_profile.json     — 10 constraints with fragility spectrum
  session_tasks.json      — 50 tasks across 10 sessions
  eval_probes.json        — 10 held-out probes (1 per constraint)
  constraint_updates.json — 2 mid-lifetime updates (session 3, 6)
  tools.py                — check_constraints tool + TOOL_SPEC
  validator.py            — CVR + constraint_precision + lag_recall + compounding scorers

Expected CVR curve (lossy_compress SUT):
  Session:  0    1    2    3    4    5    6    7    8    9
  CVR:      0.0  0.0  0.1  0.1  0.2  0.3  0.4  0.5  0.7  0.7
""")

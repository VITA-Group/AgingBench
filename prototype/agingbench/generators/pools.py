"""Shared entity and value pools for programmatic scenario generators.

Sources:
- Person names: US Census most common first/last names (public domain)
- Product names: Magento sample data + fictional e-commerce products
- Company names: Fictional, inspired by real industry categories
- Place names: Real-world POIs (OpenStreetMap, public domain)
- Tech stack: Real framework/tool names from StackOverflow survey
- Food/cuisine: USDA FoodData Central categories + common cuisine types
"""

from __future__ import annotations

import random

# ---------------------------------------------------------------------------
# Person names (100+ combinations)
# ---------------------------------------------------------------------------
FIRST_NAMES = [
    "Sarah", "Marcus", "Priya", "James", "Elena", "David", "Anika", "Tom",
    "Emily", "Ravi", "Sofia", "Lin", "Jing", "Carlos", "Fatima", "Noah",
    "Olivia", "Liam", "Ava", "Ethan", "Mia", "Lucas", "Isabella", "Mason",
    "Zara", "Owen", "Chloe", "Kai", "Luna", "Finn", "Aria", "Leo",
    "Nadia", "River", "Sage", "Rowan", "Quinn", "Blake", "Skylar", "Asher",
    "Yuki", "Hana", "Wei", "Soren", "Petra", "Nikolai", "Mei", "Arjun",
    "Ingrid", "Rafael", "Amara", "Dante", "Leila", "Henrik", "Vera", "Felix",
    "Rosa", "Ivan", "Camille", "Oscar", "Thea", "Hugo", "Elise", "Axel",
]

LAST_NAMES = [
    "Chen", "Rivera", "Patel", "Kim", "Vasquez", "Nakamura", "O'Brien",
    "Zhang", "Sharma", "Delacroix", "Okonkwo", "Hoffman", "Wu", "Garcia",
    "Singh", "Johansson", "Park", "Nguyen", "Fischer", "Santos", "Lee",
    "Anderson", "Tanaka", "Mendez", "Larsson", "Krishnaswamy", "Burke",
    "Chowdhury", "Petrov", "Bergström", "Moreau", "Takahashi", "Volkov",
    "Reyes", "Müller", "Kovacs", "Osei", "Ferreira", "Antonov", "Lindgren",
    "Yamamoto", "Dubois", "Henriksen", "Kowalski", "Sundaram", "Mbeki",
    "Carlsson", "Popov", "Hashimoto", "Ruiz",
]

# ---------------------------------------------------------------------------
# Product names (fictional e-commerce products)
# ---------------------------------------------------------------------------
PRODUCT_NAMES = [
    "Quest Lumaflex Band", "AeroGlide Pro X1", "NovaPulse Tracker",
    "Dash Digital Watch", "Sprite Foam Yoga Brick", "Crown Summit Backpack",
    "Zenith Wireless Earbuds", "Apex Running Vest", "Fusion Smart Scale",
    "Titan Grip Resistance Set", "Solstice Meditation Cushion",
    "Cascade Hydration Pack", "Ember Trail Headlamp", "Vertex Climbing Chalk",
    "Aura Recovery Roller", "Bolt Speed Jump Rope", "Summit Edge Trekking Pole",
    "Echo Pulse Heart Monitor", "Nimbus Sleep Mask Pro", "Terra Grip Trail Shoe",
    "Horizon Cycling Jersey", "Drift Kayak Paddle", "Prism UV Sunglasses",
    "Peak Performance Gloves", "Orbit Fitness Tracker", "Zephyr Wind Jacket",
    "Crest Wave Board Short", "Flux Compression Sleeve", "Atlas Camp Stove",
    "Vortex Swim Goggles", "Onyx Yoga Mat Premium", "Breeze Running Cap",
    "Stratos Altitude Watch", "Ripple Water Bottle", "Blaze Training Shoe",
    "Mystic Balance Board", "Canyon Hiking Boot", "Frost Ice Pack Set",
    "Luna Night Light Band", "Phoenix Recovery Wrap",
]

# ---------------------------------------------------------------------------
# Company / vendor names (fictional)
# ---------------------------------------------------------------------------
COMPANY_NAMES = [
    "ShieldStack Security", "CloudVault Systems", "DataPeak Analytics",
    "Vertex Solutions", "NexaBridge Technologies", "PulsePoint Cloud",
    "CipherNet Labs", "StreamForge IO", "Quantum Relay Inc",
    "IronGate Hosting", "BlueShift Data", "TerraScale Computing",
    "ArcLight Networks", "ZenithOps Platform", "NovaCrest Software",
    "PrismView Analytics", "Catalyst DevTools", "Meridian Infrastructure",
    "Cobalt Automation", "Starline Consulting", "Aegis Compliance",
    "Ember Logic Systems", "TrueNorth Monitoring", "Axiom Cloud Services",
    "Radiant Security Group", "Keystone Platform", "Helix Integration",
    "Beacon Data Solutions", "Nimbus Infrastructure", "Forge DevOps",
]

# ---------------------------------------------------------------------------
# Place names (real-world POIs from OpenStreetMap, public domain)
# ---------------------------------------------------------------------------
RESTAURANT_NAMES = [
    "La Prima Espresso", "Choolaah Indian BBQ", "Bella Notte Trattoria",
    "Stack'd Burgers", "Golden Lotus Thai", "Sakura Sushi House",
    "El Mercado Cantina", "Blue Moon Bistro", "The Rustic Table",
    "Crescent Moon Café", "Harbor Light Seafood", "Fireside Grill",
    "Jade Palace Chinese", "Olive Tree Mediterranean", "Maple & Rye",
    "Copper Kettle Kitchen", "The Wandering Chef", "Sunset Terrace",
    "North Star Diner", "Wildflower Bakery", "Stone Bridge Pub",
    "Saffron Spice House", "Cloud Nine Rooftop", "The Iron Skillet",
    "Pearl Street Oyster Bar", "Bamboo Garden", "Red Lantern Noodles",
    "The Gilded Fork", "Harvest Moon Farm Table", "Ember & Oak",
]

UNIVERSITY_NAMES = [
    "Carnegie Mellon University", "MIT", "Stanford University",
    "UC Berkeley", "Georgia Tech", "University of Michigan",
    "University of Washington", "Cornell University", "Princeton University",
    "Columbia University", "Yale University", "Brown University",
    "Duke University", "Northwestern University", "Rice University",
]

PARK_NAMES = [
    "Acadia National Park", "Shenandoah National Park", "Olympic National Park",
    "Great Smoky Mountains", "Yellowstone National Park", "Glacier National Park",
    "Rocky Mountain National Park", "Zion National Park", "Grand Canyon",
    "Joshua Tree National Park", "Everglades National Park", "Badlands National Park",
]

CITY_NAMES = [
    "Pittsburgh", "Boston", "San Francisco", "Seattle", "Austin", "Denver",
    "Portland", "Nashville", "Minneapolis", "Atlanta", "Chicago", "Miami",
    "Philadelphia", "Charlotte", "Indianapolis", "Columbus", "Detroit",
    "Salt Lake City", "Tampa", "Raleigh",
]

# ---------------------------------------------------------------------------
# Tech stack (real framework/tool names, StackOverflow survey)
# ---------------------------------------------------------------------------
TECH_FRAMEWORKS = [
    "PostgreSQL", "Redis", "FastAPI", "Django", "Flask", "SQLAlchemy",
    "Next.js", "React", "Vue.js", "Tailwind CSS", "Docker", "Kubernetes",
    "Terraform", "GitHub Actions", "GitLab CI", "Datadog", "PagerDuty",
    "Grafana", "Prometheus", "Nginx", "Celery", "RabbitMQ", "Kafka",
    "Elasticsearch", "MongoDB", "DynamoDB", "TimescaleDB", "ClickHouse",
]

LIBRARY_PAIRS = [
    # (preferred, alternative) — for S5 convention constraints
    ("pendulum", "datetime"), ("structlog", "logging"), ("httpx", "requests"),
    ("pydantic", "dataclasses"), ("arrow", "datetime"), ("attrs", "dataclasses"),
    ("orjson", "json"), ("uvloop", "asyncio"), ("aiohttp", "requests"),
    ("rich", "print"), ("typer", "argparse"), ("pytest", "unittest"),
]

# ---------------------------------------------------------------------------
# Code-domain confusable APIs/methods (for S4 software-engineering
# interference). Near-twin entity names that SHARE a stem but have DISTINCT
# behavior — the confusion is in the codebase itself, not a bolted-on business
# term. Each entry: a shared stem, two confusable names with contrasting
# behaviors, and a forced-choice probe whose gold is name_a (distractor name_b).
# Used by S4 when PressureConfig.confusable_topic_matched is set.
# ---------------------------------------------------------------------------
CODE_CONFUSABLE_PAIRS = [
    {"stem": "by-tag", "name_a": "filter_by_tag", "name_b": "sort_by_tag",
     "desc_a": "returns ONLY the subset of notes matching the tag",
     "desc_b": "returns ALL notes reordered by their first tag",
     "probe_question": "Which function returns only the subset of notes matching a tag — filter_by_tag or sort_by_tag? Reply with the exact function name."},
    {"stem": "dict", "name_a": "from_dict", "name_b": "to_dict",
     "desc_a": "parses a dict into a model instance",
     "desc_b": "serializes a model instance into a dict",
     "probe_question": "Which method parses a dict into a model instance — from_dict or to_dict? Reply with the exact method name."},
    {"stem": "by-id", "name_a": "get_by_id", "name_b": "get_by_ids",
     "desc_a": "returns a single record for one id",
     "desc_b": "returns a list of records for a list of ids",
     "probe_question": "Which function returns a single record for one id — get_by_id or get_by_ids? Reply with the exact function name."},
    {"stem": "save", "name_a": "save", "name_b": "save_all",
     "desc_a": "persists a single record",
     "desc_b": "persists every record in the batch",
     "probe_question": "Which method persists only a single record — save or save_all? Reply with the exact method name."},
    {"stem": "find", "name_a": "find_one", "name_b": "find_all",
     "desc_a": "returns the first matching record or None",
     "desc_b": "returns a list of all matching records",
     "probe_question": "Which method returns just the first matching record — find_one or find_all? Reply with the exact method name."},
    {"stem": "serialize", "name_a": "serialize", "name_b": "deserialize",
     "desc_a": "converts an object to its stored representation",
     "desc_b": "reconstructs an object from its stored representation",
     "probe_question": "Which function converts an object to its stored representation — serialize or deserialize? Reply with the exact function name."},
    {"stem": "update", "name_a": "update", "name_b": "upsert",
     "desc_a": "modifies an existing record and fails if absent",
     "desc_b": "modifies an existing record or inserts it if absent",
     "probe_question": "Which method fails if the record is absent — update or upsert? Reply with the exact method name."},
    {"stem": "delete", "name_a": "delete", "name_b": "delete_all",
     "desc_a": "removes a single record by id",
     "desc_b": "removes every record in the table",
     "probe_question": "Which command removes only a single record by id — delete or delete_all? Reply with the exact command name."},
    {"stem": "config", "name_a": "load_config", "name_b": "reload_config",
     "desc_a": "reads the config once at startup",
     "desc_b": "re-reads the config file at runtime, discarding cached values",
     "probe_question": "Which function reads the config once at startup — load_config or reload_config? Reply with the exact function name."},
    {"stem": "note", "name_a": "add_note", "name_b": "append_note",
     "desc_a": "creates a new note with a fresh id",
     "desc_b": "appends text to the body of an existing note",
     "probe_question": "Which command creates a brand-new note with a fresh id — add_note or append_note? Reply with the exact command name."},
    {"stem": "count", "name_a": "count", "name_b": "count_distinct",
     "desc_a": "returns the total number of rows",
     "desc_b": "returns the number of unique values in a column",
     "probe_question": "Which function returns the total number of rows — count or count_distinct? Reply with the exact function name."},
    {"stem": "list", "name_a": "list_notes", "name_b": "list_tags",
     "desc_a": "lists every stored note",
     "desc_b": "lists the distinct tags across all notes",
     "probe_question": "Which command lists every stored note — list_notes or list_tags? Reply with the exact command name."},
]

# ---------------------------------------------------------------------------
# Food and cuisine (USDA categories + common types)
# ---------------------------------------------------------------------------
CUISINE_TYPES = [
    "Italian", "Japanese", "Mexican", "Indian", "Thai", "Chinese",
    "Mediterranean", "Korean", "Vietnamese", "French", "Ethiopian",
    "Greek", "Turkish", "Peruvian", "Brazilian", "Lebanese", "Moroccan",
    "Spanish", "German", "Caribbean",
]

FOOD_ITEMS = [
    "pasta", "sushi", "tacos", "curry", "pad thai", "dumplings",
    "falafel", "bibimbap", "pho", "croissant", "pizza", "ramen",
    "burrito", "steak", "salmon", "tofu stir-fry", "lamb kebab",
    "fish and chips", "paella", "risotto",
]

DIETARY_RESTRICTIONS = [
    "shellfish allergy", "gluten intolerance", "lactose intolerance",
    "nut allergy", "celiac disease", "soy allergy", "egg allergy",
    "vegetarian", "vegan", "pescatarian", "low sodium", "low sugar",
    "kosher", "halal", "dairy-free",
]

# ---------------------------------------------------------------------------
# Budget categories (US Bureau of Labor Statistics)
# ---------------------------------------------------------------------------
BUDGET_CATEGORIES = [
    "dining", "shopping", "subscriptions", "groceries", "transportation",
    "entertainment", "fitness", "clothing", "electronics", "travel",
    "home_maintenance", "personal_care", "education", "gifts", "utilities",
]

# ---------------------------------------------------------------------------
# Project components (for S1/S3 project narratives)
# ---------------------------------------------------------------------------
PROJECT_COMPONENTS = [
    "API Gateway", "Auth Service", "Payment Processor", "Search Engine",
    "Notification Service", "User Management", "Analytics Pipeline",
    "Data Warehouse", "CDN Layer", "Cache Layer", "Message Queue",
    "Rate Limiter", "Load Balancer", "Monitoring Stack", "CI/CD Pipeline",
    "Database Migration", "Feature Flags", "A/B Testing Framework",
    "Logging Infrastructure", "Security Scanner",
]

# V2 extension (opt-in via PressureConfig.project_components_pool_version=2).
# Includes the original 20 entries plus 30 more spanning observability,
# data-platform, security, ops, and developer-tools domains. The V1 pool is
# preserved unchanged so existing seeds reproduce bit-for-bit.
PROJECT_COMPONENTS_V2 = PROJECT_COMPONENTS + [
    "Service Mesh", "Message Bus", "Event Store", "Time-Series DB",
    "Vector Store", "Graph Database", "Object Storage", "CDN Edge Node",
    "Build Pipeline", "Artifact Registry", "Container Registry",
    "Kubernetes Operator", "Helm Chart Renderer", "Service Catalog",
    "Identity Provider", "Authorization Service", "Audit Logger",
    "Compliance Scanner", "Vulnerability Scanner", "License Compliance",
    "Data Loss Prevention", "Threat Intelligence Feed", "Honeypot",
    "Bug Bounty Triage", "Incident Manager", "On-Call Pager",
    "Status Dashboard", "Synthetic Monitor", "Real User Monitor",
    "Distributed Tracer",
]
assert len(PROJECT_COMPONENTS_V2) == 50, (
    f"PROJECT_COMPONENTS_V2 expected 50 entries; got {len(PROJECT_COMPONENTS_V2)}"
)

# V3 extension (opt-in via PressureConfig.project_components_pool_version=3).
# Includes the V2 50 entries plus 50 RESEARCH-INFRASTRUCTURE components so the
# S1 "Research Literature Aging" framing is actually carried by the content,
# not just the scenario label. The new entries are systems/pipelines/tools
# that research labs build and operate, paired with the existing operational
# templates (latency, throughput, deployment, audit, etc.). V1/V2 are
# preserved unchanged for reproducibility.
PROJECT_COMPONENTS_V3 = PROJECT_COMPONENTS_V2 + [
    # ML training infrastructure (12)
    "Training Pipeline", "Distributed Trainer", "Checkpoint Manager",
    "Gradient Accumulator", "Mixed-Precision Trainer",
    "DataLoader Service", "Tokenization Pipeline", "Embedding Cache",
    "LoRA Adapter Registry", "Quantization Service",
    "Distillation Pipeline", "Optimizer State Sharder",
    # Experimentation infrastructure (12)
    "Experiment Tracker", "Hyperparameter Search Engine", "Ablation Runner",
    "A/B Experiment Service", "Multi-Armed Bandit Service",
    "Pre-Registration Vault", "Reproducibility Sandbox", "Seed Manager",
    "Result Aggregator", "Plot Generator", "Metric Logger",
    "Diagnostic Probe Suite",
    # Evaluation infrastructure (8)
    "Benchmark Harness", "Eval Harness", "Leaderboard Service",
    "Human Eval Platform", "LLM-as-Judge Service", "Calibration Calculator",
    "Statistical Significance Suite", "Cross-Validation Splitter",
    # Data + model infrastructure (10)
    "Model Registry", "Dataset Catalog", "Data Lineage Tracker",
    "Annotation Platform", "Active Learning Loop",
    "Synthetic Data Generator", "Prompt Library", "RAG Index",
    "Reward Model Service", "Foundation Model Cache",
    # Research process tooling (8)
    "Citation Graph Service", "Literature Mining Tool", "arXiv Watcher",
    "Conference Submission Tracker", "Peer Review Workflow",
    "Notebook Server", "Jupyter Hub", "Paper Drafting Assistant",
]
assert len(PROJECT_COMPONENTS_V3) == 100, (
    f"PROJECT_COMPONENTS_V3 expected 100 entries; got {len(PROJECT_COMPONENTS_V3)}"
)


def get_project_components(version: int = 1) -> list[str]:
    """Resolve the PROJECT_COMPONENTS pool for a given version.

    Version 1 (default) returns the original 20-entry generic infra pool.
    Version 2 returns the 50-entry extended generic pool.
    Version 3 returns the 100-entry pool that adds RESEARCH-INFRASTRUCTURE
    components (training pipelines, eval harnesses, experiment trackers,
    arXiv watchers, etc.) — making the S1 "Research Literature Aging"
    framing genuinely carried by the content.

    Versions are append-only; rng calls with same seed + same version
    produce identical output. Switching versions changes output.
    """
    return {
        1: PROJECT_COMPONENTS,
        2: PROJECT_COMPONENTS_V2,
        3: PROJECT_COMPONENTS_V3,
    }[version]

PROJECT_MILESTONES = [
    "Alpha Release", "Beta Launch", "Public Preview", "GA Release",
    "Phase 1 Completion", "Phase 2 Kickoff", "Security Audit",
    "Performance Review", "Compliance Certification", "Scale Test",
    "User Acceptance Testing", "Production Rollout",
]

# ---------------------------------------------------------------------------
# Code entities for S4/S5 (domain model names)
# ---------------------------------------------------------------------------
CODE_ENTITIES = [
    "User", "Product", "Order", "Session", "Config", "Payment",
    "Invoice", "Customer", "Inventory", "Shipment", "Notification",
    "Subscription", "Review", "Comment", "Category", "Tag",
    "Permission", "Role", "Audit", "Report",
]

CODE_FIELDS = {
    "User": [("name", "str"), ("email", "str"), ("age", "int")],
    "Product": [("title", "str"), ("price", "float"), ("stock", "int")],
    "Order": [("user_id", "str"), ("total", "float"), ("status", "str")],
    "Session": [("token", "str"), ("user_id", "str"), ("expires_at", "str")],
    "Config": [("key", "str"), ("value", "str"), ("env", "str")],
    "Payment": [("amount", "float"), ("method", "str"), ("order_id", "str")],
    "Invoice": [("number", "str"), ("total", "float"), ("due_date", "str")],
    "Customer": [("name", "str"), ("email", "str"), ("tier", "str")],
    "Inventory": [("product_id", "str"), ("quantity", "int"), ("warehouse", "str")],
    "Shipment": [("order_id", "str"), ("carrier", "str"), ("tracking", "str")],
    "Notification": [("user_id", "str"), ("message", "str"), ("read", "bool")],
    "Subscription": [("plan", "str"), ("price", "float"), ("active", "bool")],
    "Review": [("product_id", "str"), ("rating", "int"), ("text", "str")],
    "Comment": [("author", "str"), ("body", "str"), ("post_id", "str")],
    "Category": [("name", "str"), ("parent_id", "str"), ("slug", "str")],
    "Tag": [("name", "str"), ("color", "str"), ("description", "str")],
    "Permission": [("name", "str"), ("resource", "str"), ("action", "str")],
    "Role": [("name", "str"), ("level", "int"), ("permissions", "str")],
    "Audit": [("action", "str"), ("user_id", "str"), ("timestamp", "str")],
    "Report": [("title", "str"), ("data", "str"), ("format", "str")],
}

# ---------------------------------------------------------------------------
# Occasions and contexts (for S2 task templates)
# ---------------------------------------------------------------------------
OCCASIONS = [
    "Friday", "birthday", "anniversary", "team celebration", "date night",
    "family gathering", "holiday", "casual weekend", "business lunch",
    "farewell dinner", "graduation", "promotion celebration",
]

# ---------------------------------------------------------------------------
# Value generators
# ---------------------------------------------------------------------------

def ensure_non_round(value: int, rng: random.Random) -> int:
    """Adjust value to avoid multiples of 5/10/25/50/100."""
    while value % 5 == 0:
        value += rng.randint(1, 4)
    return value


def random_dollar(rng: random.Random, lo: int = 100, hi: int = 50000) -> int:
    """Random non-round dollar amount."""
    return ensure_non_round(rng.randint(lo, hi), rng)


def random_percent(rng: random.Random, lo: float = 50.0, hi: float = 99.9) -> str:
    """Random percentage with one decimal, avoiding round numbers."""
    val = rng.uniform(lo, hi)
    while abs(val - round(val)) < 0.05:
        val = rng.uniform(lo, hi)
    return f"{val:.1f}%"


def random_distance_km(rng: random.Random, lo: int = 10, hi: int = 2000) -> int:
    """Random non-round distance in km."""
    return ensure_non_round(rng.randint(lo, hi), rng)


def random_date(rng: random.Random, year: int = 2026) -> str:
    """Random date in the given year."""
    month = rng.randint(1, 12)
    day = rng.randint(1, 28)
    return f"{year}-{month:02d}-{day:02d}"


def random_count(rng: random.Random, lo: int = 10, hi: int = 10000) -> int:
    """Random non-round count."""
    return ensure_non_round(rng.randint(lo, hi), rng)


def random_latency_ms(rng: random.Random, lo: int = 10, hi: int = 500) -> int:
    """Random non-round latency in ms."""
    return ensure_non_round(rng.randint(lo, hi), rng)


def random_person(rng: random.Random) -> str:
    """Random full name from pools."""
    return f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"


def sample_unique(pool: list, k: int, rng: random.Random) -> list:
    """Sample k unique items from pool."""
    return rng.sample(pool, min(k, len(pool)))


# ------------------------------------------------------------------ interference

CONFUSABLE_TERMS = {
    "budget": {
        "domains": ["dining", "vendor", "project", "marketing"],
        "value_ranges": [(100, 500), (10000, 100000), (50000, 500000), (5000, 20000)],
    },
    "timeline": {
        "domains": ["delivery", "project", "contract", "hiring"],
        "value_ranges": [(7, 90), (30, 365), (180, 730), (14, 60)],
    },
    "approval": {
        "domains": ["budget", "vendor", "hiring", "security"],
        "value_ranges": [(1000, 50000), (5000, 200000), (50000, 150000), (1000, 10000)],
    },
    "capacity": {
        "domains": ["storage", "compute", "network", "team"],
        "value_ranges": [(100, 10000), (8, 256), (100, 10000), (5, 50)],
    },
    "threshold": {
        "domains": ["performance", "quality", "security", "compliance"],
        "value_ranges": [(80, 100), (90, 100), (95, 100), (85, 100)],
    },
    # ---- Expanded interference terms (P4) ----
    "deadline": {
        "domains": ["project", "contract", "legal", "hiring"],
        "value_ranges": [(30, 365), (90, 730), (60, 180), (14, 90)],
    },
    "headcount": {
        "domains": ["engineering", "marketing", "operations", "support"],
        "value_ranges": [(3, 25), (2, 15), (5, 40), (4, 20)],
    },
    "project_code": {
        "domains": ["engineering", "marketing", "operations", "finance"],
        "value_ranges": [(1000, 9999), (1000, 9999), (1000, 9999), (1000, 9999)],
    },
    "vendor_id": {
        "domains": ["cloud", "hardware", "consulting", "SaaS"],
        "value_ranges": [(10000, 99999), (10000, 99999), (10000, 99999), (10000, 99999)],
    },
    "utilization": {
        "domains": ["compute", "storage", "network", "office_space"],
        "value_ranges": [(50, 99), (30, 95), (40, 98), (20, 85)],
    },
    "discount": {
        "domains": ["vendor", "subscription", "bulk_order", "loyalty"],
        "value_ranges": [(5, 30), (10, 50), (15, 40), (3, 20)],
    },
    "response_time": {
        "domains": ["API", "support_ticket", "incident", "deployment"],
        "value_ranges": [(50, 500), (1, 72), (1, 24), (5, 120)],
    },
    "priority_level": {
        "domains": ["bug_triage", "feature_request", "security", "compliance"],
        "value_ranges": [(1, 5), (1, 5), (1, 5), (1, 5)],
    },
    "retention_rate": {
        "domains": ["customer", "employee", "subscriber", "partner"],
        "value_ranges": [(60, 99), (70, 95), (50, 98), (75, 100)],
    },
    "allocation": {
        "domains": ["R_and_D", "infrastructure", "training", "contingency"],
        "value_ranges": [(10000, 200000), (20000, 500000), (5000, 50000), (10000, 100000)],
    },
}

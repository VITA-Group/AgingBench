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

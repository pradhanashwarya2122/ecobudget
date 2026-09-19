"""Builds data/tasks_seed.json: 15 hand-written seed tasks over the corpus
built by build_seed_corpus.py. Answers below are derived directly from the
same sourced facts as the corpus passages -- ground truth, never used by
retrieval/stopping logic, only by evaluation.

DEPRECATED as a regeneration tool (2026-09-12). data/tasks_seed.json is now the
canonical source of truth and contains many seeds added after this script
(T16+ from earlier expansions, T42-T48 from the train-coverage fix, T49+ from
the Phase 1 scale-up). Re-running this script would OVERWRITE all of them. Kept
only as the historical record of the original 15 seeds. Add new seeds via
scripts/expand_seed_tasks.py (idempotent append), not here.

Each task's `decomposed_requirements` uses (entity, attribute) pairs that
must exist in the corpus; scripts/validate_corpus_task_coverage.py checks
this automatically.

Run: python scripts/build_seed_tasks.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SEED_TASKS = [
    {
        "id": "T1",
        "question": "What is the battery capacity of the iPhone 15?",
        "decomposed_requirements": [{"entity": "iPhone 15", "attribute": "battery"}],
        "expected_answer": "3349 mAh",
        "topic": "phones",
        "difficulty": "easy",
        "answer_type": "single_fact",
    },
    {
        "id": "T2",
        "question": "What is the starting price of the Samsung Galaxy S24?",
        "decomposed_requirements": [{"entity": "Samsung Galaxy S24", "attribute": "price"}],
        "expected_answer": "$799",
        "topic": "phones",
        "difficulty": "easy",
        "answer_type": "single_fact",
    },
    {
        "id": "T3",
        "question": "Which phone has the higher display refresh rate: the iPhone 15 or the Samsung Galaxy S24?",
        "decomposed_requirements": [
            {"entity": "iPhone 15", "attribute": "display_refresh_rate"},
            {"entity": "Samsung Galaxy S24", "attribute": "display_refresh_rate"},
        ],
        "expected_answer": "Samsung Galaxy S24 (up to 120Hz vs iPhone 15's 60Hz)",
        "topic": "phones",
        "difficulty": "medium",
        "answer_type": "comparison",
    },
    {
        "id": "T4",
        "question": "Compare the iPhone 15 and Samsung Galaxy S24 on price and battery capacity.",
        "decomposed_requirements": [
            {"entity": "iPhone 15", "attribute": "price"},
            {"entity": "iPhone 15", "attribute": "battery"},
            {"entity": "Samsung Galaxy S24", "attribute": "price"},
            {"entity": "Samsung Galaxy S24", "attribute": "battery"},
        ],
        "ground_truth": {
            "required_facts": [
                "iPhone 15 starts at $799",
                "Samsung Galaxy S24 starts at $799",
                "iPhone 15 has a 3349 mAh battery",
                "Samsung Galaxy S24 has a 4000 mAh battery",
            ],
            "match_threshold": 1.0,
        },
        "topic": "phones",
        "difficulty": "hard",
        "answer_type": "comparison",
    },
    {
        "id": "T5",
        "question": "What is the starting price of the MacBook Air 13 M3?",
        "decomposed_requirements": [{"entity": "MacBook Air 13 M3", "attribute": "price"}],
        "expected_answer": "$1,099",
        "topic": "laptops",
        "difficulty": "easy",
        "answer_type": "single_fact",
    },
    {
        "id": "T6",
        "question": "What is the battery life of the Dell XPS 13 (2024)?",
        "decomposed_requirements": [{"entity": "Dell XPS 13 2024", "attribute": "battery"}],
        "expected_answer": "roughly 14 hours of continuous video playback in independent testing",
        "topic": "laptops",
        "difficulty": "easy",
        "answer_type": "single_fact",
    },
    {
        "id": "T7",
        "question": "Which laptop is lighter: the MacBook Air 13 M3 or the Dell XPS 13 (2024)?",
        "decomposed_requirements": [
            {"entity": "MacBook Air 13 M3", "attribute": "weight"},
            {"entity": "Dell XPS 13 2024", "attribute": "weight"},
        ],
        "expected_answer": "Dell XPS 13 2024 (about 2.6 lb vs MacBook Air's 2.7 lb)",
        "topic": "laptops",
        "difficulty": "medium",
        "answer_type": "comparison",
    },
    {
        "id": "T8",
        "question": "Compare the MacBook Air 13 M3 and the Dell XPS 13 (2024) on price and weight.",
        "decomposed_requirements": [
            {"entity": "MacBook Air 13 M3", "attribute": "price"},
            {"entity": "MacBook Air 13 M3", "attribute": "weight"},
            {"entity": "Dell XPS 13 2024", "attribute": "price"},
            {"entity": "Dell XPS 13 2024", "attribute": "weight"},
        ],
        "ground_truth": {
            "required_facts": [
                "MacBook Air 13 M3 starts at $1,099",
                "MacBook Air 13 M3 weighs about 2.7 pounds",
                "Dell XPS 13 2024 starts at $1,299",
                "Dell XPS 13 2024 weighs about 2.6 pounds",
            ],
            "match_threshold": 1.0,
        },
        "topic": "laptops",
        "difficulty": "hard",
        "answer_type": "comparison",
    },
    {
        "id": "T9",
        "question": "What is the best time of year to visit Jaipur?",
        "decomposed_requirements": [{"entity": "Jaipur", "attribute": "best_time_to_visit"}],
        "expected_answer": "October to March",
        "topic": "travel",
        "difficulty": "easy",
        "answer_type": "single_fact",
    },
    {
        "id": "T10",
        "question": "What is the top attraction to visit in Kyoto?",
        "decomposed_requirements": [{"entity": "Kyoto", "attribute": "top_attraction"}],
        "expected_answer": "Fushimi Inari Shrine",
        "topic": "travel",
        "difficulty": "easy",
        "answer_type": "single_fact",
    },
    {
        "id": "T11",
        "question": "Compare Jaipur and Kyoto on typical daily travel budget.",
        "decomposed_requirements": [
            {"entity": "Jaipur", "attribute": "daily_budget"},
            {"entity": "Kyoto", "attribute": "daily_budget"},
        ],
        "ground_truth": {
            "required_facts": [
                "Jaipur is cheaper in its April-June off-season",
                "Kyoto ranges from about $40-60/day backpacker to $264+/day luxury",
            ],
            "match_threshold": 1.0,
        },
        "topic": "travel",
        "difficulty": "hard",
        "answer_type": "comparison",
    },
    {
        "id": "T12",
        "question": "Which city is recommended for a shorter trip: Jaipur or Kyoto?",
        "decomposed_requirements": [
            {"entity": "Jaipur", "attribute": "typical_trip_length"},
            {"entity": "Kyoto", "attribute": "typical_trip_length"},
        ],
        "expected_answer": "Both are similar (Jaipur 2-4 days, Kyoto 3-4 days), Jaipur slightly shorter at the low end",
        "topic": "travel",
        "difficulty": "medium",
        "answer_type": "comparison",
    },
    {
        "id": "T13",
        "question": "What chip powers the iPhone 15?",
        "decomposed_requirements": [{"entity": "iPhone 15", "attribute": "chip"}],
        "expected_answer": "Apple A16 Bionic",
        "topic": "phones",
        "difficulty": "easy",
        "answer_type": "single_fact",
    },
    {
        "id": "T14",
        "question": "What processor options are available for the Dell XPS 13 (2024)?",
        "decomposed_requirements": [{"entity": "Dell XPS 13 2024", "attribute": "chip"}],
        "expected_answer": "Intel Core Ultra 200V-series or Qualcomm Snapdragon X Elite",
        "topic": "laptops",
        "difficulty": "medium",
        "answer_type": "single_fact",
    },
    {
        "id": "T15",
        "question": "Compare the main camera resolution of the iPhone 15 and the Samsung Galaxy S24.",
        "decomposed_requirements": [
            {"entity": "iPhone 15", "attribute": "camera"},
            {"entity": "Samsung Galaxy S24", "attribute": "camera"},
        ],
        "expected_answer": "Samsung Galaxy S24 has a 50MP main camera vs iPhone 15's 48MP",
        "topic": "phones",
        "difficulty": "medium",
        "answer_type": "comparison",
    },

    # --- Hotels (travel) ---
    {
        "id": "T16",
        "question": "How many rooms does The Oberoi Rajvilas Jaipur have?",
        "decomposed_requirements": [{"entity": "Oberoi Rajvilas Jaipur", "attribute": "room_count"}],
        "expected_answer": "71",
        "topic": "travel",
        "difficulty": "easy",
        "answer_type": "single_fact",
    },
    {
        "id": "T17",
        "question": "Which Jaipur hotel is more affordable per night: The Oberoi Rajvilas Jaipur or Ibis Jaipur City Centre?",
        "decomposed_requirements": [
            {"entity": "Oberoi Rajvilas Jaipur", "attribute": "price_per_night"},
            {"entity": "Ibis Jaipur City Centre", "attribute": "price_per_night"},
        ],
        "expected_answer": "Ibis Jaipur City Centre (roughly $28-40/night vs Oberoi Rajvilas's $260+/night)",
        "topic": "travel",
        "difficulty": "medium",
        "answer_type": "comparison",
    },
    {
        "id": "T18",
        "question": "Compare The Oberoi Rajvilas Jaipur and Ibis Jaipur City Centre on price per night and amenities.",
        "decomposed_requirements": [
            {"entity": "Oberoi Rajvilas Jaipur", "attribute": "price_per_night"},
            {"entity": "Oberoi Rajvilas Jaipur", "attribute": "amenities"},
            {"entity": "Ibis Jaipur City Centre", "attribute": "price_per_night"},
            {"entity": "Ibis Jaipur City Centre", "attribute": "amenities"},
        ],
        "ground_truth": {
            "required_facts": [
                "Oberoi Rajvilas is roughly $260-800+ per night",
                "Ibis Jaipur City Centre is roughly $28-40 per night",
                "Oberoi Rajvilas has a spa, pool, tennis courts, and multiple restaurants",
                "Ibis Jaipur City Centre has a rooftop pool, gym, and one restaurant",
            ],
            "match_threshold": 1.0,
        },
        "topic": "travel",
        "difficulty": "hard",
        "answer_type": "comparison",
    },

    # --- Wireless earbuds (product selection) ---
    {
        "id": "T19",
        "question": "What is the battery life of the Sony WF-1000XM5?",
        "decomposed_requirements": [{"entity": "Sony WF-1000XM5", "attribute": "battery_life"}],
        "expected_answer": "about 8 hours with ANC on",
        "topic": "electronics",
        "difficulty": "easy",
        "answer_type": "single_fact",
    },
    {
        "id": "T20",
        "question": "Which earbuds have better noise cancellation: AirPods Pro 2 or Sony WF-1000XM5?",
        "decomposed_requirements": [
            {"entity": "AirPods Pro 2", "attribute": "noise_cancellation"},
            {"entity": "Sony WF-1000XM5", "attribute": "noise_cancellation"},
        ],
        "expected_answer": "Sony WF-1000XM5, per independent reviewer testing",
        "topic": "electronics",
        "difficulty": "medium",
        "answer_type": "comparison",
    },
    {
        "id": "T21",
        "question": "Compare AirPods Pro 2 and Sony WF-1000XM5 on price and battery life.",
        "decomposed_requirements": [
            {"entity": "AirPods Pro 2", "attribute": "price"},
            {"entity": "AirPods Pro 2", "attribute": "battery_life"},
            {"entity": "Sony WF-1000XM5", "attribute": "price"},
            {"entity": "Sony WF-1000XM5", "attribute": "battery_life"},
        ],
        "ground_truth": {
            "required_facts": [
                "AirPods Pro 2 costs $249",
                "Sony WF-1000XM5 costs $299.99",
                "AirPods Pro 2 lasts about 6 hours with ANC on",
                "Sony WF-1000XM5 lasts about 8 hours with ANC on",
            ],
            "match_threshold": 1.0,
        },
        "topic": "electronics",
        "difficulty": "hard",
        "answer_type": "comparison",
    },

    # --- Buildings (specification lookup) ---
    {
        "id": "T22",
        "question": "How tall is the Burj Khalifa?",
        "decomposed_requirements": [{"entity": "Burj Khalifa", "attribute": "height"}],
        "expected_answer": "828 meters (2,717 feet)",
        "topic": "buildings",
        "difficulty": "easy",
        "answer_type": "single_fact",
    },
    {
        "id": "T23",
        "question": "Which building is taller: the Burj Khalifa or the Empire State Building?",
        "decomposed_requirements": [
            {"entity": "Burj Khalifa", "attribute": "height"},
            {"entity": "Empire State Building", "attribute": "height"},
        ],
        "expected_answer": "Burj Khalifa (828m vs 381m roof height)",
        "topic": "buildings",
        "difficulty": "easy",
        "answer_type": "comparison",
    },
    {
        "id": "T24",
        "question": "Compare the Burj Khalifa and the Empire State Building on height and completion year.",
        "decomposed_requirements": [
            {"entity": "Burj Khalifa", "attribute": "height"},
            {"entity": "Burj Khalifa", "attribute": "completion_year"},
            {"entity": "Empire State Building", "attribute": "height"},
            {"entity": "Empire State Building", "attribute": "completion_year"},
        ],
        "ground_truth": {
            "required_facts": [
                "Burj Khalifa is 828 meters tall",
                "Empire State Building is 381 meters tall at the roof",
                "Burj Khalifa was completed in 2010",
                "Empire State Building was completed in 1931",
            ],
            "match_threshold": 1.0,
        },
        "topic": "buildings",
        "difficulty": "medium",
        "answer_type": "comparison",
    },

    # --- Mountains (specification lookup) ---
    {
        "id": "T25",
        "question": "What is the elevation of Mount Everest?",
        "decomposed_requirements": [{"entity": "Mount Everest", "attribute": "elevation"}],
        "expected_answer": "8,849 meters (29,032 feet)",
        "topic": "geography",
        "difficulty": "easy",
        "answer_type": "single_fact",
    },
    {
        "id": "T26",
        "question": "Which mountain was first climbed earlier: Mount Everest or K2?",
        "decomposed_requirements": [
            {"entity": "Mount Everest", "attribute": "first_ascent_year"},
            {"entity": "K2", "attribute": "first_ascent_year"},
        ],
        "expected_answer": "Mount Everest (1953 vs K2's 1954)",
        "topic": "geography",
        "difficulty": "medium",
        "answer_type": "comparison",
    },
    {
        "id": "T27",
        "question": "Compare Mount Everest and K2 on elevation and location.",
        "decomposed_requirements": [
            {"entity": "Mount Everest", "attribute": "elevation"},
            {"entity": "Mount Everest", "attribute": "location"},
            {"entity": "K2", "attribute": "elevation"},
            {"entity": "K2", "attribute": "location"},
        ],
        "ground_truth": {
            "required_facts": [
                "Everest is 8,849 meters, in the Himalayas on the Nepal-China border",
                "K2 is 8,611 meters, in the Karakoram Range on the Pakistan-China border",
            ],
            "match_threshold": 1.0,
        },
        "topic": "geography",
        "difficulty": "medium",
        "answer_type": "comparison",
    },

    # --- Countries (multi-attribute research) ---
    {
        "id": "T28",
        "question": "What is the capital of Japan?",
        "decomposed_requirements": [{"entity": "Japan", "attribute": "capital"}],
        "expected_answer": "Tokyo",
        "topic": "countries",
        "difficulty": "easy",
        "answer_type": "single_fact",
    },
    {
        "id": "T29",
        "question": "Which country has a larger population: India or Japan?",
        "decomposed_requirements": [
            {"entity": "India", "attribute": "population"},
            {"entity": "Japan", "attribute": "population"},
        ],
        "expected_answer": "India (about 1.4-1.48 billion vs Japan's ~123 million)",
        "topic": "countries",
        "difficulty": "easy",
        "answer_type": "comparison",
    },
    {
        "id": "T30",
        "question": "Compare India and Japan on population and land area.",
        "decomposed_requirements": [
            {"entity": "India", "attribute": "population"},
            {"entity": "India", "attribute": "area"},
            {"entity": "Japan", "attribute": "population"},
            {"entity": "Japan", "attribute": "area"},
        ],
        "ground_truth": {
            "required_facts": [
                "India has roughly 1.4-1.48 billion people",
                "Japan has roughly 123 million people",
                "India covers about 3,287,263 sq km",
                "Japan covers about 377,930-377,975 sq km",
            ],
            "match_threshold": 1.0,
        },
        "topic": "countries",
        "difficulty": "medium",
        "answer_type": "comparison",
    },

    # --- Running shoes (decision making) ---
    {
        "id": "T31",
        "question": "What is the price of the Nike Pegasus 41?",
        "decomposed_requirements": [{"entity": "Nike Pegasus 41", "attribute": "price"}],
        "expected_answer": "about $130-145",
        "topic": "running_shoes",
        "difficulty": "easy",
        "answer_type": "single_fact",
    },
    {
        "id": "T32",
        "question": "Which running shoe is lighter: the Nike Pegasus 41 or the Hoka Clifton 9?",
        "decomposed_requirements": [
            {"entity": "Nike Pegasus 41", "attribute": "weight"},
            {"entity": "Hoka Clifton 9", "attribute": "weight"},
        ],
        "expected_answer": "Hoka Clifton 9 (about 9.7 oz vs Pegasus 41's 10.0 oz)",
        "topic": "running_shoes",
        "difficulty": "medium",
        "answer_type": "comparison",
    },
    {
        "id": "T33",
        "question": "Compare the Nike Pegasus 41 and Hoka Clifton 9 on cushioning and recommended use.",
        "decomposed_requirements": [
            {"entity": "Nike Pegasus 41", "attribute": "cushioning"},
            {"entity": "Nike Pegasus 41", "attribute": "best_for"},
            {"entity": "Hoka Clifton 9", "attribute": "cushioning"},
            {"entity": "Hoka Clifton 9", "attribute": "best_for"},
        ],
        "ground_truth": {
            "required_facts": [
                "Pegasus 41 has moderate cushioning, suited to daily training and tempo work",
                "Clifton 9 has maximal cushioning, suited to long runs and easy mileage",
            ],
            "match_threshold": 1.0,
        },
        "topic": "running_shoes",
        "difficulty": "hard",
        "answer_type": "comparison",
    },

    # --- Electric vehicles (sustainability) ---
    {
        "id": "T34",
        "question": "What is the EPA-estimated range of the Tesla Model 3?",
        "decomposed_requirements": [{"entity": "Tesla Model 3", "attribute": "range"}],
        "expected_answer": "roughly 309-363 miles depending on trim",
        "topic": "vehicles",
        "difficulty": "easy",
        "answer_type": "single_fact",
    },
    {
        "id": "T35",
        "question": "Which EV has the longer range: the Tesla Model 3 or the Nissan Leaf?",
        "decomposed_requirements": [
            {"entity": "Tesla Model 3", "attribute": "range"},
            {"entity": "Nissan Leaf", "attribute": "range"},
        ],
        "expected_answer": "Tesla Model 3 (309-363 miles vs Leaf's 259-303 miles)",
        "topic": "vehicles",
        "difficulty": "medium",
        "answer_type": "comparison",
    },
    {
        "id": "T36",
        "question": "Compare the Tesla Model 3 and Nissan Leaf on price and battery capacity.",
        "decomposed_requirements": [
            {"entity": "Tesla Model 3", "attribute": "price"},
            {"entity": "Tesla Model 3", "attribute": "battery_capacity"},
            {"entity": "Nissan Leaf", "attribute": "price"},
            {"entity": "Nissan Leaf", "attribute": "battery_capacity"},
        ],
        "ground_truth": {
            "required_facts": [
                "Tesla Model 3 starts around $37,000-44,000",
                "Nissan Leaf starts around $29,990",
                "Tesla Model 3 battery ranges from about 60 to 82 kWh",
                "Nissan Leaf uses a 75 kWh battery (52 kWh entry trim)",
            ],
            "match_threshold": 1.0,
        },
        "topic": "vehicles",
        "difficulty": "hard",
        "answer_type": "comparison",
    },

    # --- General question types ---
    {
        "id": "T37",
        "question": "Does the Tesla Model 3 have a longer EPA-estimated range than the Nissan Leaf?",
        "decomposed_requirements": [
            {"entity": "Tesla Model 3", "attribute": "range"},
            {"entity": "Nissan Leaf", "attribute": "range"},
        ],
        "expected_answer": "Yes",
        "topic": "vehicles",
        "difficulty": "easy",
        "answer_type": "yes_no",
    },
    {
        "id": "T38",
        "question": "What amenities does The Oberoi Rajvilas Jaipur offer?",
        "decomposed_requirements": [
            {"entity": "Oberoi Rajvilas Jaipur", "attribute": "amenities"},
        ],
        "ground_truth": {
            "required_facts": [
                "outdoor pool",
                "full-service spa",
                "fitness center",
                "restaurants",
                "tennis courts",
            ],
            "match_threshold": 0.6,
        },
        "topic": "travel",
        "difficulty": "medium",
        "answer_type": "list",
    },
    {
        "id": "T39",
        "question": "What are the capital and official languages of India?",
        "decomposed_requirements": [
            {"entity": "India", "attribute": "capital"},
            {"entity": "India", "attribute": "official_language"},
        ],
        "ground_truth": {
            "required_facts": [
                "India's capital is New Delhi",
                "Hindi and English are official languages at the union level",
            ],
            "match_threshold": 1.0,
        },
        "topic": "countries",
        "difficulty": "medium",
        "answer_type": "multi_part",
    },
    {
        "id": "T40",
        "question": "How do I change a flat car tire?",
        "decomposed_requirements": [{"entity": "tire", "attribute": "procedure"}],
        "ground_truth": {
            "required_facts": [
                "secure the vehicle and loosen the lug nuts",
                "raise the vehicle at the jacking point",
                "remove the flat tire and install the spare",
                "lower the vehicle and tighten the lug nuts",
            ],
            "match_threshold": 1.0,
        },
        "topic": "vehicles",
        "difficulty": "hard",
        "answer_type": "procedure",
    },
    {
        "id": "T41",
        "question": "Summarize the premise of Mamma Mia! (2008).",
        "decomposed_requirements": [{"entity": "Mamma Mia", "attribute": "summary"}],
        "ground_truth": (
            "On a Greek island, Sophie secretly invites three former lovers of "
            "her mother Donna to her wedding because one may be her father, "
            "reviving old relationships."
        ),
        "topic": "film",
        "difficulty": "medium",
        "answer_type": "narrative",
    },
]


def main() -> None:
    out_path = ROOT / "data" / "tasks_seed.json"
    out_path.write_text(json.dumps(SEED_TASKS, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(SEED_TASKS)} seed tasks to {out_path}")


if __name__ == "__main__":
    main()

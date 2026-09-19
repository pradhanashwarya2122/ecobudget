"""Builds data/corpus.jsonl: a real (not placeholder) seed passage corpus.

DEPRECATED as a regeneration tool (2026-09-12). data/corpus.jsonl is now the
canonical source of truth: it carries passages and embeddings added after this
script was written (the decomposer train-coverage fix + Phase 1 scale-up).
Re-running this script would OVERWRITE those and drop the embeddings. Kept only
as the historical record of the original seed. Add new passages via
scripts/expand_seed_tasks.py (idempotent append), not here.


Facts below are paraphrased in our own words from web-sourced material
(product spec pages / review sites for the phones and laptops, travel-guide
sites for the cities), gathered 2026-09-09. Each passage carries its
source_url for provenance. This is a SEED corpus (30 passages across 6
entities) -- Phase 1's target is 200-500. Per the plan's review addendum,
scaling the remaining volume should draw on an existing open-source passage
dataset (e.g. a subset of MS MARCO or Natural Questions) adapted into this
same Passage schema, rather than manually sourcing everything by hand.

Run: python scripts/build_seed_corpus.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ml_retriever.types import Passage  # noqa: E402

# Each entry: (entity, attribute, text, source_url)
RAW_PASSAGES = [
    # --- iPhone 15 ---
    ("iPhone 15", "price",
     "The iPhone 15 starts at $799 for the 128GB configuration, with 256GB "
     "and 512GB options priced higher.",
     "https://www.tomsguide.com/reviews/iphone-15"),
    ("iPhone 15", "battery",
     "The iPhone 15 has a 3349 mAh battery, rated by Apple for up to about "
     "20 hours of video playback.",
     "https://www.gsmarena.com/apple_iphone_15-12559.php"),
    ("iPhone 15", "display_refresh_rate",
     "The iPhone 15 has a 6.1-inch OLED display with a 60Hz refresh rate; "
     "it does not get the 120Hz ProMotion display reserved for Pro models.",
     "https://www.techspecs.info/apple-iphone-15/"),
    ("iPhone 15", "weight",
     "The iPhone 15 weighs about 171 grams.",
     "https://iphonescompare.com/iphone-15-specs/"),
    ("iPhone 15", "chip",
     "The iPhone 15 runs on Apple's A16 Bionic chip, a 4nm processor also "
     "used in the previous year's iPhone 14 Pro models.",
     "https://iphonescompare.com/iphone-15-specs/"),
    ("iPhone 15", "camera",
     "The iPhone 15 has a 48MP main camera and is the first standard "
     "(non-Pro) iPhone to include the Dynamic Island.",
     "https://iphonescompare.com/iphone-15-specs/"),

    # --- Samsung Galaxy S24 ---
    ("Samsung Galaxy S24", "price",
     "The Samsung Galaxy S24 starts at $799 for the base 128GB, 8GB RAM "
     "configuration in the US.",
     "https://tomsguide.com/news/samsung-galaxy-s24"),
    ("Samsung Galaxy S24", "battery",
     "The Samsung Galaxy S24 has a 4,000 mAh battery, supporting 25W wired "
     "charging and 15W wireless charging.",
     "https://crackthecable.com/specs/galaxys24"),
    ("Samsung Galaxy S24", "display_refresh_rate",
     "The Samsung Galaxy S24 has a 6.2-inch Dynamic AMOLED 2X display with "
     "an adaptive refresh rate that ranges from 1Hz to 120Hz.",
     "https://www.techspecs.info/samsung-galaxy-s24/"),
    ("Samsung Galaxy S24", "weight",
     "The Samsung Galaxy S24 weighs 168 grams.",
     "https://www.devicespecifications.com/en/model/20d85e13"),
    ("Samsung Galaxy S24", "chip",
     "The Samsung Galaxy S24 uses a Qualcomm Snapdragon 8 Gen 3 chip in "
     "most markets (Exynos 2400 in some regions), paired with 8GB of RAM "
     "in the base configuration.",
     "https://www.devicespecifications.com/en/model/20d85e13"),
    ("Samsung Galaxy S24", "camera",
     "The Samsung Galaxy S24 has a 50MP main camera as part of a triple "
     "rear camera system.",
     "https://www.backmarket.com/en-us/c/samsung/samsung-s24-tech-specs"),

    # --- MacBook Air 13 (M3, 2024) ---
    ("MacBook Air 13 M3", "price",
     "The MacBook Air 13-inch M3 starts at $1,099 for the base "
     "configuration with 8GB of RAM and 256GB of storage.",
     "https://www.laptopmag.com/laptops/macbooks/macbook-air-13-inch-m3"),
    ("MacBook Air 13 M3", "battery",
     "The MacBook Air 13-inch M3 has a 52.6 watt-hour battery, rated by "
     "Apple for up to 18 hours of movie playback or up to 15 hours of "
     "wireless web browsing.",
     "https://www.macworld.com/article/2000691/13-15-inch-macbook-air-m3-release-date-specs-rumors-html.html"),
    ("MacBook Air 13 M3", "weight",
     "The MacBook Air 13-inch M3 weighs about 2.7 pounds (roughly 1.24 kg).",
     "https://everymac.com/systems/apple/macbook-air/specs/macbook-air-m3-8-core-cpu-8-core-gpu-13-2024-specs.html"),
    ("MacBook Air 13 M3", "display",
     "The MacBook Air 13-inch M3 has a 13.6-inch Liquid Retina display at "
     "2560x1664 resolution with 500 nits of peak brightness.",
     "https://invgate.com/itdb/macbook-air-m3-13"),
    ("MacBook Air 13 M3", "chip",
     "The MacBook Air 13-inch M3 uses Apple's M3 chip, an 8-core-CPU, "
     "8-core-GPU processor built on a 3nm process.",
     "https://invgate.com/itdb/macbook-air-m3-13"),

    # --- Dell XPS 13 (2024) ---
    ("Dell XPS 13 2024", "price",
     "The Dell XPS 13 (2024), in its Snapdragon X Elite configuration, "
     "starts at $1,299 with 16GB of RAM and 512GB of storage.",
     "https://tech.yahoo.com/computing/articles/dell-xps-13-2024-review-201327653.html"),
    ("Dell XPS 13 2024", "battery",
     "Independent lab testing measured the Dell XPS 13 (2024) at roughly "
     "14 hours of continuous video playback; Dell's own marketing for a "
     "different 2026 configuration cites up to 17 hours of streaming "
     "battery life.",
     "https://www.pcworld.com/article/2330227/dell-xps-13-2024-review.html"),
    ("Dell XPS 13 2024", "weight",
     "The Dell XPS 13 (2024) weighs about 2.6 pounds (roughly 1.18 kg).",
     "https://www.laptopmag.com/laptops/windows-laptops/dell-xps-13-9350"),
    ("Dell XPS 13 2024", "display",
     "The Dell XPS 13 (2024) has a 13.4-inch display at 1920x1200 "
     "resolution.",
     "https://tech.yahoo.com/computing/articles/dell-xps-13-2024-review-201327653.html"),
    ("Dell XPS 13 2024", "chip",
     "The Dell XPS 13 (2024) has been sold with either an Intel Core Ultra "
     "200V-series processor or, in a separate 2024 model, a Qualcomm "
     "Snapdragon X Elite chip.",
     "https://www.laptopmag.com/laptops/windows-laptops/dell-xps-13-9350"),

    # --- Jaipur ---
    ("Jaipur", "best_time_to_visit",
     "The best time to visit Jaipur is October to March, when "
     "temperatures are cooler and more comfortable for sightseeing; this "
     "is also the peak tourist season, with festivals like Diwali and the "
     "Jaipur Literature Festival.",
     "https://www.jaipurunveiled.com/best-time-to-visit-jaipur/"),
    ("Jaipur", "daily_budget",
     "Jaipur is cheaper to visit in its off-season, April to June, when "
     "summer heat pushes temperatures above 40C but hotel and tour "
     "prices drop for budget-conscious travelers.",
     "https://www.tripzygo.com/blogs/best-time-to-visit-jaipur"),
    ("Jaipur", "top_attraction",
     "Amber Fort is commonly recommended as the top attraction to start a "
     "Jaipur visit, alongside the City Palace and Hawa Mahal.",
     "https://www.eliteindiatour.com/blog-details/jaipur-travel-guide"),
    ("Jaipur", "typical_trip_length",
     "Most Jaipur itineraries suggest two to four days to cover the "
     "city's major sights.",
     "https://crystalindiaholidays.com/best-time-to-visit-jaipur/"),

    # --- Kyoto ---
    ("Kyoto", "best_time_to_visit",
     "Kyoto is most scenic in spring (March-May) for cherry blossoms and "
     "in autumn for foliage, though winter (December-February) is "
     "cheaper for budget travelers.",
     "https://www.getyourguide.com/explorer/kyoto-ttd96826/best-time-to-visit-kyoto/"),
    ("Kyoto", "daily_budget",
     "Daily budgets for Kyoto vary widely by travel style: roughly "
     "$40-60 per day for a backpacker, $120-200 per day for a mid-range "
     "traveler, and $264 or more per day for a luxury traveler.",
     "https://www.budgetyourtrip.com/japan/kyoto"),
    ("Kyoto", "top_attraction",
     "Fushimi Inari Shrine, known for its thousands of vermilion torii "
     "gates, is free to enter and open 24 hours, and is frequently listed "
     "as Kyoto's top attraction.",
     "https://tourismattractions.net/japan/kyoto-on-a-budget"),
    ("Kyoto", "typical_trip_length",
     "Most first-time visitors spend three to four days in Kyoto.",
     "https://tourismattractions.net/japan/kyoto-tourism-attractions"),

    # --- Hotels (travel / product selection) ---
    ("Oberoi Rajvilas Jaipur", "price_per_night",
     "Nightly rates at The Oberoi Rajvilas Jaipur vary widely by room type "
     "and booking source, roughly $260 to $800 or more per night, with one "
     "site reporting an average around $385 in its cheapest month.",
     "https://www.kayak.co.in/Jaipur-Hotels-The-Oberoi-Rajvilas.56545.ksp"),
    ("Oberoi Rajvilas Jaipur", "rating",
     "The Oberoi Rajvilas Jaipur is a 5-star hotel with a 9.5 (\"Wonderful\") "
     "guest rating based on several hundred reviews.",
     "https://www.kayak.com/Jaipur-Hotels-The-Oberoi-Rajvilas.56545.ksp"),
    ("Oberoi Rajvilas Jaipur", "amenities",
     "Amenities include an outdoor pool, a full-service spa, a fitness "
     "center, two to three restaurants, and outdoor tennis courts, set "
     "across 32 acres of gardens.",
     "https://www.kayak.co.in/Jaipur-Hotels-The-Oberoi-Rajvilas.56545.ksp"),
    ("Oberoi Rajvilas Jaipur", "room_count",
     "The hotel has 71 rooms and suites.",
     "https://www.travelweekly.com/Hotels/Jaipur-India/The-Oberoi-Rajvilas-Hotel-p3715012"),

    ("Ibis Jaipur City Centre", "price_per_night",
     "Nightly rates at Ibis Jaipur City Centre are roughly $28-40 per "
     "night (about INR 2,300-3,300).",
     "https://www.kayak.co.in/Jaipur-Hotels-Ibis-Jaipur.308125.ksp"),
    ("Ibis Jaipur City Centre", "rating",
     "Ibis Jaipur City Centre is a 4-star hotel rated around 4.3 out of 5 "
     "based on tens of thousands of guest reviews.",
     "https://www.justdial.com/Jaipur/Ibis-Hotel-Near-Civil-Lines-Metro-Station-Civil-Lines/0141PX141-X141-130417171409-D7G1_BZDET"),
    ("Ibis Jaipur City Centre", "amenities",
     "Amenities include a rooftop pool, a 24-hour gym, an on-site "
     "restaurant serving Indian cuisine, and a rooftop terrace.",
     "https://www.expedia.com/Jaipur-District-Hotels-Ibis-Jaipur-Civil-Lines-Hotel.h5932432.Hotel-Information"),
    ("Ibis Jaipur City Centre", "room_count",
     "The hotel has approximately 140 rooms.",
     "https://www.booking.com/hotel/in/ibis-jaipur.html"),

    # --- Wireless earbuds (product selection) ---
    ("AirPods Pro 2", "price",
     "The AirPods Pro 2 (USB-C model) are priced at $249.",
     "https://www.tomsguide.com/face-off/airpods-pro-2-vs-sony-wf-1000xm4"),
    ("AirPods Pro 2", "battery_life",
     "The AirPods Pro 2 offer about 6 hours of listening per charge with "
     "noise cancellation on, plus roughly 30 hours total using the "
     "charging case.",
     "https://price.review/articles/airpods-pro-2-vs-sony-wf1000xm5"),
    ("AirPods Pro 2", "noise_cancellation",
     "The AirPods Pro 2 have strong active noise cancellation, though "
     "independent reviewers rated Sony's WF-1000XM5 as slightly stronger "
     "across the frequency spectrum.",
     "https://recordingnow.com/blog/sony-wf-1000xm5-vs-airpods-pro-2/"),
    ("AirPods Pro 2", "codec_support",
     "The AirPods Pro 2 support the AAC audio codec; they do not support "
     "LDAC or aptX.",
     "https://price.review/articles/airpods-pro-2-vs-sony-wf1000xm5"),

    ("Sony WF-1000XM5", "price",
     "The Sony WF-1000XM5 are priced at $299.99.",
     "https://appleinsider.com/inside/airpods-pro-2/vs/airpods-pro-vs-sony-wf-1000xm5----compared"),
    ("Sony WF-1000XM5", "battery_life",
     "The Sony WF-1000XM5 offer about 8 hours of listening per charge "
     "with noise cancellation on, longer than the AirPods Pro 2's rated "
     "single-charge life.",
     "https://price.review/articles/airpods-pro-2-vs-sony-wf1000xm5"),
    ("Sony WF-1000XM5", "noise_cancellation",
     "Independent reviewers rate the Sony WF-1000XM5's active noise "
     "cancellation as slightly stronger than the AirPods Pro 2's across "
     "the frequency spectrum.",
     "https://recordingnow.com/blog/sony-wf-1000xm5-vs-airpods-pro-2/"),
    ("Sony WF-1000XM5", "codec_support",
     "The Sony WF-1000XM5 support the LDAC high-resolution codec in "
     "addition to AAC and SBC, an advantage over AirPods Pro 2's AAC-only "
     "support.",
     "https://price.review/articles/airpods-pro-2-vs-sony-wf1000xm5"),

    # --- Buildings (specification lookup) ---
    ("Burj Khalifa", "height",
     "The Burj Khalifa has an architectural height of 828 meters (2,717 "
     "feet), or 829.8 meters (2,722 feet) to its very tip.",
     "https://en.wikipedia.org/wiki/Burj_Khalifa"),
    ("Burj Khalifa", "floors",
     "The Burj Khalifa has 163 floors above ground.",
     "https://www.britannica.com/topic/Burj-Khalifa"),
    ("Burj Khalifa", "completion_year",
     "The Burj Khalifa topped out in January 2009 and formally opened in "
     "January 2010.",
     "https://en.wikipedia.org/wiki/Burj_Khalifa"),
    ("Burj Khalifa", "architect",
     "The Burj Khalifa was designed by Adrian Smith of the Chicago-based "
     "firm Skidmore, Owings & Merrill.",
     "https://www.britannica.com/topic/Burj-Khalifa"),

    ("Empire State Building", "height",
     "The Empire State Building's roof is 381 meters (1,250 feet) tall; "
     "including its antenna, the tip reaches 443 meters (1,454 feet).",
     "https://en.wikipedia.org/wiki/Empire_State_Building"),
    ("Empire State Building", "floors",
     "The Empire State Building has 102 floors.",
     "https://www.britannica.com/topic/Empire-State-Building"),
    ("Empire State Building", "completion_year",
     "The Empire State Building formally opened on May 1, 1931.",
     "https://www.britannica.com/topic/Empire-State-Building"),
    ("Empire State Building", "architect",
     "The Empire State Building was designed by the firm Shreve, Lamb and "
     "Harmon.",
     "https://en.wikipedia.org/wiki/Empire_State_Building"),

    # --- Mountains (specification lookup) ---
    ("Mount Everest", "elevation",
     "Mount Everest stands 8,849 meters (29,032 feet) above sea level, "
     "the highest point on Earth.",
     "https://www.nestadventure.com/blog/everest-vs-k2/"),
    ("Mount Everest", "location",
     "Mount Everest is located in the Himalayas, on the border between "
     "Nepal and the Tibet Autonomous Region of China.",
     "https://www.nestadventure.com/blog/everest-vs-k2/"),
    ("Mount Everest", "first_ascent_year",
     "Mount Everest was first summited on May 29, 1953, by Sir Edmund "
     "Hillary and Tenzing Norgay.",
     "https://www.nestadventure.com/blog/everest-vs-k2/"),

    ("K2", "elevation",
     "K2 stands 8,611 meters (28,251 feet) above sea level, the "
     "second-highest mountain on Earth after Everest.",
     "https://en.wikipedia.org/wiki/K2"),
    ("K2", "location",
     "K2 is located in the Karakoram Range, on the border between "
     "Pakistan-administered and China-administered territory.",
     "https://www.britannica.com/place/K2"),
    ("K2", "first_ascent_year",
     "K2 was first summited on July 31, 1954, by Achille Compagnoni and "
     "Lino Lacedelli.",
     "https://en.wikipedia.org/wiki/K2"),

    # --- Countries (multi-attribute research) ---
    ("India", "population",
     "India's population is estimated at roughly 1.4 to 1.48 billion "
     "people as of 2026, making it the world's most populous country.",
     "https://www.worldometers.info/world-population/india-population/"),
    ("India", "area",
     "India covers a total land area of about 3,287,263 square "
     "kilometers.",
     "https://worldstats.io/country/in"),
    ("India", "capital",
     "India's capital is New Delhi.",
     "https://worldstats.io/country/in"),
    ("India", "official_language",
     "Hindi and English are India's two official languages at the union "
     "level; the constitution additionally recognizes 22 scheduled "
     "regional languages.",
     "https://www.geocountries.com/india"),

    ("Japan", "population",
     "Japan's population is estimated at roughly 123 million people as "
     "of 2026.",
     "https://worldstats.io/country/jp"),
    ("Japan", "area",
     "Japan covers a total land area of about 377,930 to 377,975 square "
     "kilometers.",
     "https://en.wikipedia.org/wiki/Japan"),
    ("Japan", "capital",
     "Japan's capital is Tokyo.",
     "https://www.britannica.com/place/Japan"),
    ("Japan", "official_language",
     "Japanese is Japan's official and national language.",
     "https://www.britannica.com/place/Japan"),

    # --- Running shoes (decision making) ---
    ("Nike Pegasus 41", "price",
     "The Nike Pegasus 41 is priced around $130-145.",
     "https://www.shoemyrun.com/blog/nike-vs-hoka/"),
    ("Nike Pegasus 41", "weight",
     "The Nike Pegasus 41 weighs about 10.0 ounces (men's size 9), with "
     "a 37mm heel and 27mm forefoot stack height.",
     "https://shoefitlab.com/blog/nike-pegasus-vs-hoka-clifton"),
    ("Nike Pegasus 41", "cushioning",
     "Independent reviewers rate the Nike Pegasus 41's cushioning as "
     "moderate (around 3 out of 5), built on a responsive ReactX foam "
     "midsole.",
     "https://shoefitlab.com/blog/nike-pegasus-vs-hoka-clifton"),
    ("Nike Pegasus 41", "best_for",
     "The Nike Pegasus 41 is best suited to daily training and "
     "tempo-leaning efforts, favoring runners who prefer a firmer, more "
     "traditional ride.",
     "https://shoefitlab.com/blog/nike-pegasus-vs-hoka-clifton"),

    ("Hoka Clifton 9", "price",
     "The Hoka Clifton 9 is priced around $150.",
     "https://tomsguide.com/wellness/running/hoka-clifton-10-review"),
    ("Hoka Clifton 9", "weight",
     "The Hoka Clifton 9 weighs about 9.7 ounces (men's size 9), "
     "slightly lighter than the Nike Pegasus 41.",
     "https://shoefitlab.com/blog/nike-pegasus-vs-hoka-clifton"),
    ("Hoka Clifton 9", "cushioning",
     "Independent reviewers rate the Hoka Clifton line's cushioning as "
     "maximal (around 5 out of 5), with a high stack height for extra "
     "protection.",
     "https://shoefitlab.com/blog/nike-pegasus-vs-hoka-clifton"),
    ("Hoka Clifton 9", "best_for",
     "The Hoka Clifton 9 is best suited to long runs and easy mileage, "
     "favoring runners who prioritize joint protection over a responsive "
     "feel.",
     "https://www.runningwarehouse.com/reviews/Nike-Shoe-Reviews/nike-pegasus-41.html"),

    # --- Electric vehicles (sustainability) ---
    ("Tesla Model 3", "price",
     "The 2026 Tesla Model 3 starts around $37,000 to $44,000 depending "
     "on trim and market.",
     "https://huttignissan.com/blog/nissan-leaf-vs-tesla-model-3"),
    ("Tesla Model 3", "range",
     "The Tesla Model 3's EPA-estimated range runs roughly 309 to 363 "
     "miles depending on trim.",
     "https://www.nissanofvalencia.com/blog/nissan-leaf-vs-tesla-model-3-full-commuters-guide/"),
    ("Tesla Model 3", "battery_capacity",
     "The Tesla Model 3's battery capacity varies by trim and market, "
     "from about 60 kWh up to 82 kWh (78 kWh usable).",
     "https://www.carsguide.com.au/nissan/leaf/vs/tesla-model-3"),
    ("Tesla Model 3", "best_for",
     "The Tesla Model 3 suits buyers prioritizing maximum range, faster "
     "Supercharger-network charging, and quicker acceleration.",
     "https://www.nissanofvalencia.com/blog/nissan-leaf-vs-tesla-model-3-full-commuters-guide/"),

    ("Nissan Leaf", "price",
     "The redesigned 2026 Nissan Leaf starts around $29,990.",
     "https://www.nissanofvalencia.com/blog/nissan-leaf-vs-tesla-model-3-full-commuters-guide/"),
    ("Nissan Leaf", "range",
     "The redesigned 2026 Nissan Leaf's EPA-estimated range runs roughly "
     "259 to 303 miles depending on trim.",
     "https://www.consumerreports.org/cro/cars/models/new/nissan/leaf/overview.htm"),
    ("Nissan Leaf", "battery_capacity",
     "The redesigned 2026 Nissan Leaf uses a 75 kWh battery in most "
     "trims, with an entry-level 52 kWh version also offered.",
     "https://www.consumerreports.org/cro/cars/models/new/nissan/leaf/overview.htm"),
    ("Nissan Leaf", "best_for",
     "The Nissan Leaf suits budget-conscious buyers prioritizing a lower "
     "purchase price over maximum range.",
     "https://www.nissanofvalencia.com/blog/nissan-leaf-vs-tesla-model-3-full-commuters-guide/"),

    # --- Procedure and narrative answer support ---
    ("tire", "procedure",
     "To replace a flat car tire safely, pull over on firm ground, use the "
     "parking brake, and loosen the lug nuts before lifting the car at its "
     "specified jacking point. Remove the flat, fit the spare, lower the "
     "vehicle, then tighten the lug nuts in a cross pattern.",
     "https://www.michelinman.com/auto/auto-tips-and-advice/tire-maintenance/how-to-change-a-car-tire"),
    ("Mamma Mia", "summary",
     "In the 2008 musical Mamma Mia!, Sophie prepares to marry on a Greek "
     "island and secretly invites three men from her mother Donna's past, "
     "hoping to identify which one is her father. Their arrival unsettles "
     "Donna and revives old relationships around the wedding.",
     "https://www.universalpicturesathome.com/movies/mamma-mia-the-movie"),
]


def build() -> list[Passage]:
    passages = []
    for i, (entity, attribute, text, source_url) in enumerate(RAW_PASSAGES, start=1):
        byte_size = len(text.encode("utf-8"))
        passages.append(
            Passage(
                passage_id=f"p{i:03d}",
                text=text,
                byte_size=byte_size,
                source_url=source_url,
                embedding=None,  # filled in by scripts/embed_corpus.py
                metadata={"entity": entity, "attribute": attribute},
            )
        )
    return passages


def main() -> None:
    passages = build()
    out_path = ROOT / "data" / "corpus.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for p in passages:
            row = {
                "passage_id": p.passage_id,
                "text": p.text,
                "byte_size": p.byte_size,
                "source_url": p.source_url,
                "embedding": p.embedding,
                "metadata": p.metadata,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(passages)} passages to {out_path}")


if __name__ == "__main__":
    main()

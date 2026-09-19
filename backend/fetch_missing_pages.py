import json
import os
import requests

HEADERS = {'User-Agent': 'EcoBudgetResearchBot/1.0 (student project)'}
API_URL = "https://en.wikipedia.org/w/api.php"


def fetch_via_api(title):
    params = {
        "action": "parse",
        "page": title,
        "format": "json",
        "prop": "text",
        "redirects": 1,
    }
    r = requests.get(API_URL, params=params, headers=HEADERS, timeout=15)
    r.raise_for_status()
    data = r.json()
    if "error" in data:
        raise RuntimeError(data["error"])
    return data["parse"]["text"]["*"]


def title_from_url(url):
    # https://en.wikipedia.org/wiki/Some_Title -> "Some_Title" -> "Some Title"
    slug = url.rstrip("/").split("/wiki/")[-1]
    return slug.replace("_", " ")


with open("benchmark_tasks.json") as f:
    tasks = json.load(f)

seen = set()
for task in tasks:
    path = task["page_file"]
    if path in seen:
        continue
    seen.add(path)
    if os.path.exists(path):
        print(f"skip (exists): {path}")
        continue

    title = title_from_url(task["url"])
    print(f"fetching '{title}' -> {path}")
    try:
        html = fetch_via_api(title)
    except Exception as e:
        print(f"  FAILED: {e}")
        continue

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as out:
        out.write(html)
    print(f"  saved, {len(html)} bytes")

print("done")

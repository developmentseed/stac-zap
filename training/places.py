"""Builds the place list of the generator: training/data/places.json.

- GeoNames cities15000 (CC BY 4.0): cities with 15,000 people or more, with
  country and first-level region names and population.
- Natural Earth (public domain), from gazet's index: named rivers, lakes,
  mountain ranges, deserts, seas, islands and other physical features.

Usage (from the repository root):
  uv run --no-project --with duckdb python training/places.py
"""

import json
from pathlib import Path

import duckdb

DATA = Path(__file__).parent / "data"
NE_TYPES = (
    "river", "lake", "range/mtn", "desert", "sea", "gulf", "bay", "island", "island group", "peninsula",
    "plateau", "basin", "valley", "delta", "plain", "lowland", "tundra", "wetlands", "depression",
    "geoarea", "pen/cape", "strait", "lagoon", "fjord", "sound", "coast", "isthmus", "mountain",
    "foothills", "gorge", "archipelago", "reservoir", "playa", "glacier", "region", "peak",
)

con = duckdb.connect()
countries = {
    row[0]: row[4]
    for row in (line.split("\t") for line in (DATA / "countryInfo.txt").read_text().splitlines())
    if not row[0].startswith("#")
}
admin1 = dict(line.split("\t")[:2] for line in (DATA / "admin1CodesASCII.txt").read_text().splitlines())

cols = (
    "geonameid, name, asciiname, alternatenames, latitude, longitude, fclass, fcode, country, cc2, "
    "admin1, admin2, admin3, admin4, population, elevation, dem, timezone, modified"
).split(", ")
cities = [line.split("\t") for line in (DATA / "cities15000.txt").read_text().splitlines()]
places = []
for r in cities:
    row = dict(zip(cols, r))
    places.append({
        "id": f"gn{row['geonameid']}",
        "name": row["name"],
        "kind": "city",
        "type": row["fcode"],
        "country": countries.get(row["country"], row["country"]),
        "region": admin1.get(f"{row['country']}.{row['admin1']}"),
        "population": int(row["population"] or 0),
        "lat": float(row["latitude"]),
        "lon": float(row["longitude"]),
        "source": "geonames",
    })

ne = con.sql(
    f"""
    select id, name, subtype, country, bbox
    from 'training/data/gazet/web-demo/overture-2026-10-21.0/index.parquet'
    where source = 'natural_earth' and name is not null and subtype in {NE_TYPES}
    qualify row_number() over (partition by name, subtype order by area desc) = 1
    """
).fetchall()
for id_, name, subtype, country, bbox in ne:
    places.append({
        "id": id_,
        "name": name,
        "kind": "feature",
        "type": subtype,
        "country": country,
        "region": None,
        "population": 0,
        "lat": (bbox["ymin"] + bbox["ymax"]) / 2,
        "lon": (bbox["xmin"] + bbox["xmax"]) / 2,
        "bbox": [bbox["xmin"], bbox["ymin"], bbox["xmax"], bbox["ymax"]],
        "source": "natural_earth",
    })

(DATA / "places.json").write_text(json.dumps(places, ensure_ascii=False))
kinds = {}
for p in places:
    kinds[p["type"] if p["kind"] == "feature" else "city"] = kinds.get(p["type"] if p["kind"] == "feature" else "city", 0) + 1
print(len(places), "places", sorted(kinds.items(), key=lambda t: -t[1])[:20])

# Hex database — evaluation prompts for a testing bot

A battery of **natural-language requests**, phrased the way a new researcher would
ask them (no API names), for feeding to another (less robust) AI to test whether it
can produce correct `geoworkflow.store` code. Each prompt has an **expected
approach** (the answer key) so you can grade the bot's output.

Setup the bot should assume:
```python
from geoworkflow import open_hexdb
db = open_hexdb()
```
Real data to reference: countries by ISO3 (KEN, NGA, TZA, COD, …); cities by name
(Nairobi, Lagos, …); variables `PM25`, `odiac_co2`, `lst_day`, `lst_night`,
`landcover`; monthly 2019-01 … 2024-12 (landcover is annual, categorical).

Time grammar the bot must respect (pandas/xarray-style): **selection ≠ aggregation.**
`.sel("2020-06")` selects one month; partial patterns ("all Junes", a whole year)
require a reducer (`.mean(...)`) or the climatology idiom (`.climatology(month=6)` /
`.groupby_month().mean()`). No month-name strings — integer `month=` only.

---

## Tier 1 — single use case, directly phrased

1. "Show me all the hexagons for Nairobi." 
   → `db["KEN"]["Nairobi"].plot()` (or `.to_geodataframe()` to inspect).

2. "Pull every data value we have for the hexagon with ID `UTM32737_HQ-008188_R+031618`." 
   → `db.hexagon("UTM32737_HQ-008188_R+031618")`.

3. "What variables do we have for Lagos?" 
   → `db["NGA"]["Lagos"].variables()`.

4. "Which countries are in the database?" → `db.countries()`. 
   "Which cities do we have for Kenya?" → `db["KEN"].cities()`.

5. "Map daytime land-surface temperature in Nairobi for June 2020." 
   → `db["KEN"]["Nairobi"]["lst_day"].sel("2020-06").plot()`.

---

## Tier 2 — method + time (the core analytical asks)

6. "What's the average daytime LST across all the Junes in Nairobi? Map it." 
   → `db["KEN"]["Nairobi"]["lst_day"].climatology(month=6).plot()` 
     (equivalently `...groupby_month().mean().sel(month=6)`). **The key test: this is
     climatology, NOT `.sel("...June...")`.**

7. "Give me Nairobi's mean PM2.5 for 2020." 
   → `db["KEN"]["Nairobi"]["PM25"].mean("2020")`.

8. "What's the all-time average nighttime temperature per hexagon in Lagos?" 
   → `db["NGA"]["Lagos"]["lst_night"].mean()`.

9. "Map the hottest daytime temperature each Nairobi hexagon reached in 2021." 
   → `db["KEN"]["Nairobi"]["lst_day"].max("2021").plot()`.

10. "Export Nairobi's June-2020 PM2.5 to a CSV/GeoDataFrame for each hexagon." 
    → `db["KEN"]["Nairobi"]["PM25"].sel("2020-06").to_geodataframe()` (then `.to_csv`).

11. "Show the 2019 land cover for Nairobi, coloured properly." 
    → `db["KEN"]["Nairobi"]["landcover"].sel("2019").plot()`; colours via
      `db.legend("landcover")`. (Default statistic for landcover is `majority`.)

12. "Compare CO2 emissions across Nairobi's hexagons, averaged over the whole record." 
    → `db["KEN"]["Nairobi"]["odiac_co2"].mean().plot()`.

---

## Tier 3 — editing the database (build / extend / provenance)

13. "Rebuild the whole database from our standard recipe." 
    → CLI `geoworkflow hexdb build --recipe data/config/hexdb_recipe.yaml`
      (or `from geoworkflow.store import builder; builder.build(recipe, cfg)`).

14. "Add the city of Mombasa, Kenya to the database." 
    → `db.add_city("KEN", "Mombasa")` (or CLI `hexdb add-city --iso3 KEN --name Mombasa`).

15. "Remove Lagos from the database." 
    → `db.remove_city("NGA", "Lagos")` (CLI `hexdb remove --iso3 NGA --name Lagos`).

16. "I want to add a new metric — NDVI — to every city. How would I do that?" 
    → define a `MetricSpec` (with a `MetricSource`) and `db.add_metric(metric)` /
      `hexdb add-metric --metric-yaml ndvi.yaml`; it computes for all built cities and
      updates the recipe.

17. "What settings produced this database, and what's been added since it was built?" 
    → `geoworkflow hexdb provenance` (live `_recipe.yaml`) and `--log` (the op history).

18. "Regenerate the database at 500 m hexagons instead of 250 m." 
    → edit the recipe's grid `side_length` to 500 and `build` (a new/overwriting
      warehouse — different hex size ⇒ different GridIDs).

---

## Tier 4 — ambiguous / tricky (does the bot handle them, or hallucinate?)

19. "Map the average June temperature for Malindi." 
    → "Malindi" is **ambiguous** (KEN and TZA). Correct behaviour: surface the
      ambiguity / ask which, or use `db["KEN"].city(aggid=...)`. (Tests whether the bot
      blindly guesses a country.)

20. "Plot LST for all of June." (no year given) 
    → must use climatology (`climatology(month=6)`), **not** `.sel(...)`; a bare
      `.sel` on a non-instant should be recognized as needing a reducer.

21. "Average the land-cover class over the last few years for Nairobi." 
    → landcover is **categorical** — averaging class codes is meaningless; correct
      behaviour is to use `majority` (mode), not a numeric mean. (Tests categorical
      awareness.)

22. "What's the city-wide average PM2.5 in Nairobi for each month of 2020?" (a *time
    series*, one value per month, averaged over space) 
    → **Edge / out of scope** for the per-hex accessor (it reduces over time, not over
      hexes). A good answer notes this and reaches for the cube/`hexcube.city_mean`, or
      computes the spatial mean from `to_frame()`/`sel` results — it should NOT invent a
      nonexistent `db[...].city_mean()`.

23. "Give me PM2.5 and CO2 (only those two) for one specific hexagon in Lagos." 
    → `db.hexagon("<gridid>", variables=["PM25", "odiac_co2"])` (note: a GridID is
      needed; the bot should ask for it or explain how to find one).

24. "Find the hexagon nearest to downtown Nairobi (lon 36.82, lat -1.29) and give its data." 
    → **Edge**: nearest-by-coordinate isn't in the store accessor (it's a cube helper,
      `hexcube.nearest_cell`). A good answer flags that rather than fabricating a
      `db.nearest(...)`.

---

## How to score

- **Correct**: right country/city/variable, and — critically — the right
  *selection-vs-aggregation* choice (Tier 2/4 items 6, 20, 21 are the discriminators).
- **Partially correct**: right data, wrong time semantics (e.g. tries to `.sel("June")`).
- **Hallucination**: invents methods that don't exist (`db.city_mean`, `db.nearest`,
  month-name strings) — the Tier 4 edge cases are designed to catch this.

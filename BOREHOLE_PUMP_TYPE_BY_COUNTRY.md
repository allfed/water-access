# Handpump share of borehole users, by country

Kevin, 10 Aug 2026. Companion to `ELECTRIC_BOREHOLE_SHARE_ESTIMATE.md` (method,
verification, recommended Monte Carlo ranges). Values are the % of people whose
main drinking source is a tubewell/borehole (JMP row 57) who rely on a
**human-powered handpump** - the CIL/HEMP-resilient share. The displaced share
under HEMP is `100 - handpump %`. Full provenance (sources, URLs, quotes,
arithmetic) is in `data/original_data/borehole_pump_type_estimates.csv` and
`borehole_pump_type_research.json`.

## Researched countries (94% of urban / 97% of rural borehole users)

| Country | Borehole users | Urban handpump % | Rural handpump % | Evidence (U/R) | Verified |
|---|---:|---|---|---|---|
| India | 519.6M | 25-38 | 70-80 | expert/expert | CONFIRMED |
| Bangladesh | 138.1M | 35-65 | 88-98 | expert/inventory | PLAUSIBLE |
| Pakistan | 103.6M | 21-26 | 46-53 | survey/survey | CONFIRMED |
| Nigeria | 81.2M | 10-25 | 35-60 | inventory/inventory | PLAUSIBLE |
| Indonesia | 44.9M | 2-10 | 10-35 | survey/analogue | PLAUSIBLE |
| Uganda | 20.3M | 50-85 | 88-99 | analogue/inventory | PLAUSIBLE |
| Viet Nam | 19.7M | 1-5 | 3-15 | analogue/analogue |  |
| Myanmar | 18.1M | 15-45 | 45-80 | analogue/analogue |  |
| Malawi | 12.7M | 60-90 | 92-99 | census/inventory |  |
| Afghanistan | 11.6M | 30-65 | 85-98 | expert/survey |  |
| Ukraine | 11.5M | 0-5 | 5-20 | analogue/analogue | PLAUSIBLE |
| Nepal | 10.3M | 60-90 | 90-98 | inventory/inventory |  |
| Kenya | 9.7M | 5-30 | 35-65 | expert/inventory |  |
| Burkina Faso | 9.2M | 60-90 | 90-100 | expert/admin |  |
| Cameroon | 8.2M | 10-30 | 70-95 | survey/inventory |  |
| Ethiopia | 8.2M | 30-70 | 70-92 | expert/admin |  |
| Chad | 6.9M | 70-95 | 85-100 | survey/inventory |  |
| South Sudan | 6.5M | 70-95 | 88-100 | admin/inventory |  |
| Angola | 5.7M | 15-50 | 50-85 | expert/survey |  |
| Philippines | 5.5M | 10-40 | 35-70 | analogue/analogue |  |
| Mozambique | 5.4M | 80-100 | 90-100 | survey/survey |  |
| Guinea | 5.2M | 40-80 | 90-100 | expert/inventory |  |
| Mali | 4.7M | 40-80 | 85-100 | expert/inventory |  |
| Zambia | 4.5M | 25-60 | 85-97 | expert/admin |  |
| Congo, Democratic Republic of the | 4.4M | 15-50 | 80-100 | expert/inventory |  |
| Zimbabwe | 4.4M | 40-75 | 88-98 | survey/inventory |  |
| Ghana | 4.3M | 15-40 | 75-95 | expert/inventory |  |
| Romania | 4.3M | 0-3 | 0-10 | analogue/analogue |  |
| Cambodia | 4.0M | 20-60 | 60-90 | expert/inventory |  |
| Tanzania, United Republic of | 3.8M | 15-45 | 70-92 | survey/inventory |  |
| Uzbekistan | 1.6M | 5-25 | 20-60 | -/survey |  |
| Brazil | 1.5M | 0-8 | 0-8 | analogue/analogue |  |
| Haiti | 1.0M | 20-60 | 60-90 | -/inventory |  |
| Turkmenistan | 704k | 5-25 | 20-60 | -/analogue |  |
| France | 654k | 0-2 | 0-2 | expert/expert |  |
| Austria | 633k | 0-3 | 0-3 | analogue/analogue |  |
| Norway | 547k | 0-2 | 0-2 | admin/admin |  |
| New Zealand | 399k | 0-2 | 0-2 | analogue/analogue |  |
| Bosnia and Herzegovina | 127k | 5-12 | 5-12 | census/census |  |
| Mexico | 0k | 0-10 | 0-10 | -/analogue |  |
| Peru | 0k | 0-15 | 0-15 | -/expert |  |
| Nicaragua | 0k | 2-15 | 10-40 | -/inventory |  |
| United States of America | 0k | 0-2 | 0-2 | admin/admin |  |
| Iran, Islamic Republic of | 0k | 0-10 | 0-10 | admin/admin |  |
| Kazakhstan | 0k | 5-25 | 5-25 | expert/expert |  |
| Moldova, Republic of | 0k | 0-15 | 0-15 | expert/expert |  |

Evidence kinds: census/survey = direct pump-type split in a household source
table; inventory = national water-point stock converted to user shares (bounds,
not measurements); admin = government/regulator statistics; analogue = bracketed
on a measured neighbour (e.g. Ukraine on Bosnia's census); expert = judgment,
reasoning recorded in the JSON. "Verified" = the adversarial pass re-derived the
numbers from independently located primary sources (CONFIRMED = digit-for-digit).

## Regional defaults (remaining ~5% of borehole users)

| Default region | Urban handpump % | Rural handpump % | Countries covered |
|---|---|---|---:|
| central-asia | 5-25 | 20-60 | 2 |
| europe-cis-row | 0-5 | 2-15 | 107 |
| se-asia | 2-20 | 15-65 | 35 |
| south-asia | 25-60 | 50-95 | 3 |
| ssa-east-south | 20-55 | 75-95 | 16 |
| ssa-west-central | 15-55 | 70-95 | 17 |

Known weak spot: Northern Africa (incl. Egypt) falls under the europe-cis-row
default, which is probably too electric for the rural Nile Delta handpump
tradition - ~0.4% of global weight, flagged in the estimate doc.

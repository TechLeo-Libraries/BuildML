repo: TechLeo-Libraries/BuildML
branch: main
path: buildml

## Last sync
date: 2026-08-06T17:20:00Z

### Updated in this project
- Ported Industry redesign sheets into the live EDA App (not a tinted Teaching Studio shell).
- Cockpit, Academy, and Readiness Gates follow the numbered-spine / register / stage-board layouts from this folder.
- Shared Industry tokens remain in `buildml.eda.industry_tokens` for Static EDA + App.

## Screen map
| Screen | Repo files |
| --- | --- |
| EDA Sheet - Cockpit.dc.html | buildml/dashboard/templates/index.html, static/css/tokens.css, static/css/app.css, static/js/app.js, buildml/dashboard/sheet.py, buildml/dashboard/app.py, buildml/dashboard/charts.py |
| EDA Sheet - Academy.dc.html | buildml/dashboard/academy.py, static/js/app.js, static/css/app.css |
| EDA Sheet - Readiness Gates.dc.html | buildml/dashboard/gates.py, static/js/app.js, static/css/app.css |
| Industry DS (`_ds/industry-...`) | buildml/eda/industry_tokens.py, buildml/dashboard/static/css/tokens.css |

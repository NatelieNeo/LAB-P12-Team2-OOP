# Calora: AI Meal Planner (INF1103 Team Project, Part 2)

![CI](https://github.com/NatelieNeo/Group2_Calora/actions/workflows/ci.yml/badge.svg)

**Team:** Group 2 (LAB-P12, Team 2) · **Repo:** https://github.com/NatelieNeo/Group2_Calora.git

Calora generates a personalised 3-meal daily plan from body profile, goal, dietary restrictions and budget. Code calculates calorie/macro targets, Google Gemini proposes 10 candidate meals, and code validates, filters (dietary safety) and ranks them.

> Design justification (UML, OOP mapping, SOLID, data-flow diagram, exception matrix) is in the Engineering Report. UML sources are in `docs/uml/`.

## 1. What changed in Part 2

| Area | Part 1 | Part 2 |
|---|---|---|
| Paradigm | Procedural functions | Classes, ABCs, inheritance, composition, injection |
| New feature | n/a | **Shopping list**: items to buy vs. already in pantry (`ShoppingListBuilder`, `ShoppingList`; CLI after a plan, shown on the web plan page) |
| Advanced feature | n/a | **SQLite storage + threading**: `SqliteDataManager` (select with `CALORA_STORAGE=sqlite`); `PlanJobManager` (thread pool, web polling) and `ProgressSpinner(Thread)` in the CLI |
| DevOps | Docker, pytest | Docker, pytest, GitHub Actions CI |

## 2. Structure

```
main.py  app.py  config.py
manager/
  exceptions.py          CaloraError hierarchy
  models.py              UserProfile, MealPlanRequest, MealPlan, PlannedMeal, ShoppingList ...
  logic_manager.py       NutritionCalculator, MealRule(s), RankingCriterion(s), MealTagger, ShoppingListBuilder, LogicManager
  ai_manager.py          AIProvider (ABC), GeminiProvider, StubAIProvider, PromptBuilder, AIManager
  ai_handler_manager.py  AIHandlerManager (parse + validate)
  data_manager.py        DataManager (ABC), JsonDataManager, SqliteDataManager
  job_manager.py         PlanJobManager, PlanJob, JobStatus, ProgressSpinner
  controller.py          CaloraController, AppFactory
  io_manager.py          UserInputValidator, IOManager
templates/ static/ tests/ docs/uml/
Dockerfile  .dockerignore  .github/workflows/ci.yml  .env.example
```

## 3. Configuration

Copy `.env.example` to `.env`:

| Variable | Required | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | Yes (to run) | Gemini authentication |
| `CALORA_STORAGE` | No | `json` (default) or `sqlite` |
| `FLASK_SECRET_KEY` | No | Session secret |

`.env` is git-ignored and docker-ignored. Never commit a real key; regenerate it if it was ever exposed.

## 4. Run with Docker

```
docker build -t calora .
docker run -d --rm -p 8080:8080 --env-file .env -v "${PWD}/data:/app/data" calora        # web: http://localhost:8080
docker run --rm -it --env-file .env -v "${PWD}/data:/app/data" calora python main.py     # CLI
```

The container runs gunicorn with `-w 1 --threads 4`: job state is held in memory, so use one worker process and threads for concurrency.

## 5. Tests (no API key or network needed)

```
docker run --rm calora python -m pytest -v
# or locally
pip install -r requirements.txt && python -m pytest -v
```

340 tests cover models, nutrition maths, the safety rule, ranking, response validation, shopping list, both storage backends, threading/jobs, controller, CLI and web flows, and OOP design checks (ABCs cannot be instantiated, substitutability of providers and data managers). The AI is replaced by `StubAIProvider`.

## 6. CI

`.github/workflows/ci.yml` runs on push/PR to `main`: checkout, Python 3.11, install, `pytest`; then builds the Docker image and runs the tests inside it. No secrets are needed.

## 7. Data storage

JSON (`data/user_profiles.json`, `data/meal_plans.json`) or SQLite (`data/calora.db`). A corrupt JSON file is moved to `*.corrupt` instead of being overwritten; write failures raise `StorageError`.

## 8. Known limitations

- Budget is flagged against a plan; no search for a cheaper combination.
- Web login is username-only (no password).
- Job state is in memory and lost on restart.
- Live Gemini calls are not covered by automated tests.

## 9. Team

| Member | Student ID | Contribution |
|---|---|---|
| <name> | <id> | <role> |

Submission zip: `[LabGroup_TeamNumber_ProjectPart2Final]` with report, code, tests, git history, TeamAId file, Dockerfile; no `.env` anywhere.

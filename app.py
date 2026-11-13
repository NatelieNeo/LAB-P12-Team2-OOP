"""Flask web application for Calora's meal planner."""

import os
import logging
import threading

from dotenv import load_dotenv

from flask import Flask, redirect, render_template, request, session, url_for

from manager.controller import AppFactory
from manager.exceptions import CaloraError, InvalidInputError
from manager.io_manager import IOManager
from manager.models import MealPlanRequest, UserProfile

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(threadName)s %(message)s")
for sdk_logger in ("httpx", "httpx2", "httpcore", "httpcore2", "anthropic", "google.genai", "google_genai"):
    logging.getLogger(sdk_logger).setLevel(logging.WARNING)

HOST = "127.0.0.1"
PORT = 8080
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "calora-development-key")
io = IOManager()
_controller_lock = threading.Lock()


def get_controller():
    """Create managers lazily; tests may inject a controller in app.extensions."""
    with _controller_lock:
        if "calora_controller" not in app.extensions:
            app.extensions["calora_controller"] = AppFactory.build_controller()
        return app.extensions["calora_controller"]


@app.errorhandler(CaloraError)
def application_error(error):
    status = 400 if isinstance(error, InvalidInputError) else 503
    return render_template("error.html", message=str(error)), status


def render_form(
    profile: dict | None = None,
    message: str = "",
    action: str = "/plan",
    submit_label: str = "Build today's meal plan →",
    include_profile: bool = True,
    include_preferences: bool = True,
) -> str:
    """Render the shared profile and preferences form."""
    return render_template(
        "form.html",
        profile=profile or {},
        message=message,
        action=action,
        submit_label=submit_label,
        include_profile=include_profile,
        include_preferences=include_preferences,
    )


def build_input_record(form_data: dict[str, str]) -> dict:
    """Convert browser form values into the shape used by the project layers."""
    profile = io.read_profile_form(form_data)
    meal_request = io.read_request_form(form_data, profile)
    return {**profile.to_dict(), **meal_request.to_preferences(), "strict_budget": meal_request.strict_budget}


def generate_plan(input_record: dict) -> dict | None:
    profile = UserProfile.from_dict(input_record)
    meal_request = MealPlanRequest(profile, input_record.get("meal_preference", ""),
                                   input_record.get("meal_source", "both"),
                                   input_record.get("pantry_ingredients", []),
                                   input_record.get("daily_budget"), input_record.get("strict_budget", False))
    controller = get_controller()
    plan = controller.create_meal_plan(meal_request)
    shopping = controller.get_shopping_list(plan, meal_request)
    result = plan.to_dict()
    # Render quantity/unit labels instead of dict representations in Jinja.
    for slot, meal in plan.meals.items():
        result["meals"][slot]["pantry_ingredients_used"] = meal.ingredient_labels
    result.update(shopping_items=shopping.items, shopping_covered=shopping.covered_by_pantry,
                  disclaimer=io.allergen_notice(plan))
    return result


def profile_for_username(username: str) -> dict | None:
    profile = get_controller().get_profile(username)
    return profile.to_dict() if profile else None


def profile_form_values(profile: dict) -> dict:
    values = dict(profile)
    restrictions = profile.get("dietary_restrictions", "")
    if isinstance(restrictions, list):
        restrictions = ", ".join(restrictions)
    values["dietary_restrictions"] = restrictions
    values["budget"] = profile.get("daily_budget", "")
    values["meal_preference"] = profile.get("meal_preference", "")
    values["ingredients"] = ", ".join(profile.get("pantry_ingredients", []))
    values["gender"] = {"male": "M", "female": "F"}.get(profile.get("gender"), profile.get("gender", ""))
    values["activity_level"] = {
        "moderate": "moderately active",
        "very_active": "very active",
    }.get(profile.get("activity_level"), profile.get("activity_level", ""))
    values["meal_source"] = {"home_cooked": "in", "eat_out": "out"}.get(
        profile.get("meal_source"), profile.get("meal_source", "both")
    )
    return values


def save_or_update_profile(input_record: dict) -> None:
    username = input_record["username"]
    if profile_for_username(username) is None:
        get_controller().register_profile(UserProfile.from_dict(input_record))
    else:
        get_controller().update_profile(username, UserProfile.from_dict(input_record))


@app.get("/")
def index() -> str:
    username = session.get("username")
    if not username:
        return render_template("welcome.html")
    profile = profile_for_username(username)
    if profile is None:
        session.clear()
        return render_template("welcome.html", message="Your profile could not be found. Please sign in again.")
    return render_form(profile_form_values(profile), include_profile=False)


@app.post("/login")
def login() -> str:
    username = request.form.get("username", "").strip()
    if not username:
        return render_template("welcome.html", message="Please enter a username."), 400
    loaded = profile_for_username(username)
    session["username"] = username
    if loaded is None:
        return render_form(
            {"username": username},
            "No profile exists yet. Complete your details to create one.",
            action="/profile/create",
            submit_label="Create profile",
            include_preferences=False,
        )
    return redirect(url_for("profile"))


@app.post("/plan")
def plan() -> str:
    username = session.get("username")
    if not username:
        return redirect(url_for("index"))
    profile = profile_for_username(username)
    if profile is None:
        session.clear()
        return redirect(url_for("index"))
    form_data = profile_form_values(profile)
    form_data.update(request.form.to_dict())
    form_data["username"] = username
    input_record = build_input_record(form_data)
    result = generate_plan(input_record)
    if result is None:
        return render_template("error.html", message="Unable to generate a meal plan right now."), 503
    return render_template("plan.html", result=result, show_actions=True)


@app.post("/profile/create")
def create_profile() -> str:
    input_record = build_input_record(request.form.to_dict())
    if profile_for_username(input_record["username"]) is not None:
        return render_template("welcome.html", message="That username already exists. Please sign in.")
    get_controller().register_profile(UserProfile.from_dict(input_record))
    session["username"] = input_record["username"]
    return redirect(url_for("profile"))


@app.get("/profile")
def profile() -> str:
    username = session.get("username")
    if not username:
        return redirect(url_for("index"))
    loaded = profile_for_username(username)
    if loaded is None:
        session.clear()
        return redirect(url_for("index"))
    return render_template("profile.html", profile=profile_form_values(loaded))


@app.get("/profile/edit")
def edit_profile() -> str:
    username = session.get("username")
    loaded = profile_for_username(username) if username else None
    if loaded is None:
        session.clear()
        return redirect(url_for("index"))
    return render_form(
        profile_form_values(loaded),
        action="/profile/update",
        submit_label="Save profile changes",
        include_preferences=False,
    )


@app.post("/profile/update")
def update_profile() -> str:
    username = session.get("username")
    if not username:
        return redirect(url_for("index"))
    form_data = profile_form_values(profile_for_username(username) or {})
    form_data.update(request.form.to_dict())
    form_data["username"] = username
    input_record = build_input_record(form_data)
    if profile_for_username(input_record["username"]) is None:
        session.clear()
        return redirect(url_for("index"))
    get_controller().update_profile(input_record["username"], UserProfile.from_dict(input_record))
    return render_template("profile.html", profile=profile_form_values(input_record), message="Profile updated successfully.")


@app.get("/plans")
def plans() -> str:
    username = session.get("username")
    if not username or profile_for_username(username) is None:
        session.clear()
        return redirect(url_for("index"))
    meal_plans = [plan.to_dict() for plan in get_controller().get_history(username).values()]
    return render_template("history.html", username=username, meal_plans=meal_plans)


@app.post("/logout")
def logout() -> str:
    session.clear()
    return redirect(url_for("index"))


if __name__ == "__main__":
    print(f"Calora is running at http://{HOST}:{PORT}")
    app.run(host=HOST, port=PORT, debug=True)

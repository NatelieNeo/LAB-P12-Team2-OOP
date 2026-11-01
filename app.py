"""Flask web application for Calora's meal planner."""

import os

from flask import Flask, redirect, render_template, request, session, url_for

from manager.ai_manager import call_ai
from manager.data_manager import (
    get_meal_plans_by_user,
    load_user_profiles,
    save_meal_plan,
    save_user_profile,
    update_user_profile,
)
from manager.logic_manager import build_meal_plan

HOST = "127.0.0.1"
PORT = 8080
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "calora-development-key")


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
    restrictions = [
        item.strip()
        for item in form_data.get("dietary_restrictions", "").split(",")
        if item.strip()
    ]
    if restrictions == ["nil"]:
        restrictions = []
    pantry = [
        item.strip()
        for item in form_data.get("ingredients", "").split(",")
        if item.strip()
    ]
    gender = {"M": "male", "F": "female"}.get(form_data["gender"], form_data["gender"])
    activity_level = {
        "moderately active": "moderate",
        "very active": "very_active",
    }.get(form_data["activity_level"], form_data["activity_level"])
    meal_source = {"in": "home_cooked", "out": "eat_out"}.get(
        form_data.get("meal_source", "both"), form_data.get("meal_source", "both")
    )
    return {
        "username": form_data["username"].strip(),
        "name": form_data["name"].strip(),
        "age": int(form_data["age"]),
        "gender": gender,
        "weight_kg": float(form_data["weight_kg"]),
        "height_cm": float(form_data["height_cm"]),
        "activity_level": activity_level,
        "goal": form_data["goal"],
        "dietary_restrictions": restrictions,
        "meal_preference": form_data.get("meal_preference", "").strip(),
        "meal_source": meal_source,
        "pantry_ingredients": pantry,
        "daily_budget": float(form_data["budget"]) if form_data.get("budget") else None,
    }


def generate_plan(input_record: dict) -> dict | None:
    ai_output = call_ai(input_record)
    if ai_output is None:
        return None
    return build_meal_plan(ai_output.get("recommendations", []), input_record)


def profile_for_username(username: str) -> dict | None:
    matches = load_user_profiles("username", username.strip())
    if not matches:
        return None
    return next(iter(matches.values()))


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
        save_user_profile(input_record)
    else:
        update_user_profile(username, input_record)


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
    save_or_update_profile(input_record)
    result = generate_plan(input_record)
    if result is None:
        return render_template("error.html", message="Unable to generate a meal plan right now."), 503
    save_meal_plan(result, input_record["username"])
    return render_template("plan.html", result=result, show_actions=True)


@app.post("/profile/create")
def create_profile() -> str:
    input_record = build_input_record(request.form.to_dict())
    if profile_for_username(input_record["username"]) is not None:
        return render_template("welcome.html", message="That username already exists. Please sign in.")
    save_user_profile(input_record)
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
    update_user_profile(input_record["username"], input_record)
    return render_template("profile.html", profile=profile_form_values(input_record), message="Profile updated successfully.")


@app.get("/plans")
def plans() -> str:
    username = session.get("username")
    if not username or profile_for_username(username) is None:
        session.clear()
        return redirect(url_for("index"))
    meal_plans = list(get_meal_plans_by_user(username).values())
    return render_template("history.html", username=username, meal_plans=meal_plans)


@app.post("/logout")
def logout() -> str:
    session.clear()
    return redirect(url_for("index"))


if __name__ == "__main__":
    print(f"Calora is running at http://{HOST}:{PORT}")
    app.run(host=HOST, port=PORT, debug=True)

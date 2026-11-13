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
        for path, endpoint, view, methods in rules:
            self._app.add_url_rule(path, endpoint, view, methods=methods)

    # ---- view helpers -----------------------------------------------------------
    def _current_profile(self) -> Optional[UserProfile]:
        username = session.get("username")
        return self._controller.get_profile(username) if username else None

    def _form_values(self, profile: UserProfile) -> Dict[str, Any]:
        """Profile -> the strings the HTML form fields expect."""
        data = profile.to_dict()
        prefs = profile.preferences
        data["gender"] = {"male": "M", "female": "F"}.get(data["gender"], data["gender"])
        data["activity_level"] = self.ACTIVITY_FORM_LABELS.get(data["activity_level"], data["activity_level"])
        data["dietary_restrictions"] = ", ".join(profile.dietary_restrictions)
        data["budget"] = prefs.get("daily_budget") or ""
        data["ingredients"] = ", ".join(prefs.get("pantry_ingredients", []))
        data["meal_source"] = self.SOURCE_FORM_LABELS.get(prefs.get("meal_source", "both"), "both")
        return data

    def _plan_view(self, plan: MealPlan, shopping_list=None) -> Dict[str, Any]:
        view = plan.to_dict()
        view["disclaimer"] = self._io.allergen_notice(plan)
        view["shopping_items"] = shopping_list.items if shopping_list else []
        view["shopping_covered"] = shopping_list.covered_by_pantry if shopping_list else []
        view["recipes"] = [recipe.to_dict() for recipe in self._controller.get_recipes(plan)]
        return view

    def _render_form(self, profile=None, message="", action="/plan", submit_label="Build today's meal plan →",
                     include_profile=True, include_preferences=True, status=200):
        page = render_template(
            "form.html", profile=profile or {}, message=message, action=action,
            submit_label=submit_label, include_profile=include_profile,
            include_preferences=include_preferences)
        return page, status

    @staticmethod
    def _error_message(error: CaloraError) -> str:
        if isinstance(error, AIConnectionError):
            return "Calora could not reach the AI service. Please try again in a moment."
        if isinstance(error, AIResponseError):
            return "The AI gave an unusable answer. Please try again."
        if isinstance(error, (NoSuitableMealsError, InvalidInputError)):
            return str(error)
        if isinstance(error, StorageError):
            return "Your data could not be saved. Please try again."
        return "Unable to generate a meal plan right now."

    # ---- pages ----------------------------------------------------------------
    def index(self):
        if not session.get("username"):
            return render_template("welcome.html")
        profile = self._current_profile()
        if profile is None:
            session.clear()
            return render_template("welcome.html", message="Your profile could not be found. Please sign in again.")
        return self._render_form(self._form_values(profile), include_profile=False)

    def login(self):
        try:
            username = self._io.read_username(request.form.get("username"))
        except InvalidInputError as error:
            return render_template("welcome.html", message=str(error)), 400
        session["username"] = username
        if self._controller.get_profile(username) is None:
            return self._render_form(
                {"username": username}, "No profile exists yet. Complete your details to create one.",
                action="/profile/create", submit_label="Create profile", include_preferences=False)
        return redirect(url_for("profile"))

    def plan(self):
        profile = self._current_profile()
        if profile is None:
            session.clear()
            return redirect(url_for("index"))
        try:
            plan_request = self._io.read_request_form(request.form, profile)
        except InvalidInputError as error:
            return self._render_form(self._form_values(profile), str(error), include_profile=False, status=400)
        job_id = self._controller.submit_plan_job(plan_request)
        return render_template(
            "plan_wait.html",
            status_url=url_for("plan_status", job_id=job_id),
            result_url=url_for("plan_result", job_id=job_id))

    def _owned_job(self, job_id: str):
        job = self._controller.get_plan_job(job_id)
        if job is None or job.owner != session.get("username"):
            return None
        return job

    def plan_status(self, job_id: str):
        job = self._owned_job(job_id)
        if job is None:
            return jsonify({"status": "NOT_FOUND"}), 404
        payload = {"status": job.status.value}
        if job.status is JobStatus.FAILED:
            payload["message"] = self._error_message(job.error)
        return jsonify(payload)

    def plan_result(self, job_id: str):
        job = self._owned_job(job_id)
        if job is None:
            return redirect(url_for("index"))
        if job.status is JobStatus.FAILED:
            return render_template("error.html", message=self._error_message(job.error)), 503
        if job.status is not JobStatus.DONE:
            return render_template(
                "plan_wait.html", status_url=url_for("plan_status", job_id=job_id),
                result_url=url_for("plan_result", job_id=job_id))
        shopping_list = self._controller.get_shopping_list(job.result, job.context)
        return render_template("plan.html", result=self._plan_view(job.result, shopping_list), show_actions=True)

    def create_profile(self):
        try:
            profile = self._io.read_profile_form(request.form)
        except InvalidInputError as error:
            return self._render_form(
                request.form.to_dict(), str(error), action="/profile/create", submit_label="Create profile",
                include_preferences=False, status=400)
        if self._controller.get_profile(profile.username) is not None:
            return render_template("welcome.html", message="That username already exists. Please sign in.")
        self._controller.register_profile(profile)
        session["username"] = profile.username
        return redirect(url_for("profile"))

    def profile(self):
        profile = self._current_profile()
        if profile is None:
            session.clear()
            return redirect(url_for("index"))
        return render_template("profile.html", profile=self._profile_page_values(profile))

    @staticmethod
    def _profile_page_values(profile: UserProfile) -> Dict[str, Any]:
        data = profile.to_dict(include_preferences=False)
        data["dietary_restrictions"] = ", ".join(profile.dietary_restrictions) or "none"
        return data

    def edit_profile(self):
        profile = self._current_profile()
        if profile is None:
            session.clear()
            return redirect(url_for("index"))
        return self._render_form(
            self._form_values(profile), action="/profile/update",
            submit_label="Save profile changes", include_preferences=False)

    def update_profile(self):
        current = self._current_profile()
        if current is None:
            session.clear()
            return redirect(url_for("index"))
        try:
            updated = self._io.read_profile_form(request.form, current.username)
        except InvalidInputError as error:
            return self._render_form(
                self._form_values(current), str(error), action="/profile/update",
                submit_label="Save profile changes", include_preferences=False, status=400)
        self._controller.update_profile(current.username, updated)
        return render_template("profile.html", profile=self._profile_page_values(updated),
                               message="Profile updated successfully.")

    def plans(self):
        profile = self._current_profile()
        if profile is None:
            session.clear()
            return redirect(url_for("index"))
        history = [self._plan_view(plan) for plan in self._controller.get_history(profile.username).values()]
        return render_template("history.html", username=profile.username, meal_plans=history)

    def logout(self):
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

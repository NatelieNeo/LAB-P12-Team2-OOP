"""Web entry point: gunicorn app:app  (or python app.py for development)."""
import logging
import os
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from flask import Flask, jsonify, redirect, render_template, request, session, url_for

from manager.controller import AppFactory, CaloraController
from manager.exceptions import (
    AIConnectionError, AIResponseError, CaloraError, InvalidInputError,
    NoSuitableMealsError, StorageError,
)
from manager.io_manager import IOManager
from manager.job_manager import JobStatus
from manager.models import MealPlan, UserProfile


class WebApp:
    """Flask user interface. Like CliApp it holds no business logic: forms are
    validated by the IOManager and the work is done by the controller."""

    HOST = "127.0.0.1"
    PORT = 8080
    ACTIVITY_FORM_LABELS = {"moderate": "moderately active", "very_active": "very active"}
    SOURCE_FORM_LABELS = {"home_cooked": "in", "eat_out": "out", "both": "both"}

    def __init__(self, controller: CaloraController, io: Optional[IOManager] = None):
        self._controller = controller
        self._io = io or IOManager()
        self._app = Flask(__name__)
        self._app.secret_key = os.environ.get("FLASK_SECRET_KEY", "calora-development-key")
        self._register_routes()

    @property
    def flask_app(self) -> Flask:
        return self._app

    @staticmethod
    def create(controller: Optional[CaloraController] = None) -> Flask:
        load_dotenv()
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        return WebApp(controller or AppFactory.build_controller()).flask_app

    # ---- routes ------------------------------------------------------------------
    def _register_routes(self) -> None:
        rules = (
            ("/", "index", self.index, ["GET"]),
            ("/login", "login", self.login, ["POST"]),
            ("/plan", "plan", self.plan, ["POST"]),
            ("/plan/status/<job_id>", "plan_status", self.plan_status, ["GET"]),
            ("/plan/result/<job_id>", "plan_result", self.plan_result, ["GET"]),
            ("/profile/create", "create_profile", self.create_profile, ["POST"]),
            ("/profile", "profile", self.profile, ["GET"]),
            ("/profile/edit", "edit_profile", self.edit_profile, ["GET"]),
            ("/profile/update", "update_profile", self.update_profile, ["POST"]),
            ("/plans", "plans", self.plans, ["GET"]),
            ("/logout", "logout", self.logout, ["POST"]),
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

    
app = WebApp.create()

if __name__ == "__main__":
    print(f"Calora is running at http://{WebApp.HOST}:{WebApp.PORT}")
    app.run(host=WebApp.HOST, port=WebApp.PORT, debug=True)

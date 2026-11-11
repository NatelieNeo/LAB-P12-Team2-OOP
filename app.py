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


app = WebApp.create()

if __name__ == "__main__":
    print(f"Calora is running at http://{WebApp.HOST}:{WebApp.PORT}")
    app.run(host=WebApp.HOST, port=WebApp.PORT, debug=True)

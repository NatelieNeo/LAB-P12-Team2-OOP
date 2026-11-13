"""Terminal entry point using the same managers and selection as the web app."""
import logging

from dotenv import load_dotenv

from manager.controller import AppFactory, CaloraController
from manager.exceptions import CaloraError
from manager.io_manager import IOManager
from manager.job_manager import ProgressSpinner


class CliApp:
    """Coordinate terminal interaction; business rules stay in the managers."""

    def __init__(self, controller: CaloraController, io: IOManager | None = None):
        self._controller = controller
        self._io = io or IOManager()

    def run(self) -> None:
        io, controller = self._io, self._controller
        io.welcome()
        username = io.prompt_username()
        profile = controller.get_profile(username)
        if profile is None:
            profile = io.prompt_profile(username)
            self._controller.register_profile(profile)
            io.show_message("Profile setup complete!")

        handlers = {"1": self._create_plan, "2": self._view_history,
                    "3": self._view_profile, "4": self._update_profile,
                    "R": self._view_recipes}
        while True:
            choice = io.prompt_menu_choice()
            if choice == "5":
                io.show_message("Exiting the program. Goodbye!")
                return
            handler = handlers.get(choice)
            if handler is None:
                io.show_message("Please choose a number from 1 to 5. Enter R to view recipes.")
                continue
            profile = handler(profile)

    def _create_plan(self, profile: UserProfile) -> UserProfile:
        request = self._io.prompt_request(profile)
        spinner = self._spinner_factory()
        spinner.start()
        try:
            plan = self._controller.create_meal_plan(request)
        except CaloraError as error:
            spinner.stop()
            self._io.show_message(f"Unable to generate a meal plan: {error}")
            self._io.pause()
            return profile
        spinner.stop()
        self._io.show_plan(plan)
        shopping_list = self._controller.get_shopping_list(plan, request)
        if request.meal_source != "out":
            self._io.show_shopping_list(shopping_list)
        self._io.show_recipes(self._controller.get_recipes(plan))
        self._io.pause()
        return self._controller.get_profile(profile.username) or profile

    def _view_history(self, profile: UserProfile) -> UserProfile:
        plans = self._controller.get_history(profile.username)
        if not plans:
            self._io.show_message("No meal plans found for this user.")
        else:
            self._io.show_history(plans)
        self._io.pause()
        return profile

    def _view_recipes(self, profile: UserProfile) -> UserProfile:
        plans = self._controller.get_history(profile.username)
        found_recipes = False
        for plan_id, plan in plans.items():
            recipes = self._controller.get_recipes(plan)
            if recipes:
                self._io.show_message(f"\nMeal Plan ID: {plan_id}")
                self._io.show_recipes(recipes)
                found_recipes = True
        if not found_recipes:
            self._io.show_message("No recipes found in your saved meal plans.")
        self._io.pause()
        return profile

    def _view_profile(self, profile: UserProfile) -> UserProfile:
        self._io.show_profile(profile)
        self._io.pause()
        return profile
            controller.register_profile(profile)
        else:
            io.show_message(f"Welcome back, {profile.name}!")
        while True:
            choice = io.prompt_menu_choice()
            try:
                if choice == "1":
                    request = io.prompt_request(profile)
                    spinner = ProgressSpinner()
                    spinner.start()
                    try:
                        plan = controller.create_meal_plan(request)
                    finally:
                        spinner.stop()
                    io.show_plan(plan)
                    io.show_shopping_list(controller.get_shopping_list(plan, request))
                elif choice == "2":
                    history = controller.get_history(username)
                    if history:
                        io.show_history(history)
                    else:
                        io.show_message("No meal plans found for this user.")
                elif choice == "3":
                    io.show_profile(profile)
                elif choice == "4":
                    profile = profile.updated_with(io.prompt_profile_changes(profile))
                    controller.update_profile(username, profile)
                    io.show_profile(profile)
                elif choice == "5":
                    io.show_message("Exiting the program. Goodbye!")
                    return
                else:
                    io.show_message("Please select an option from 1 to 5.")
                    continue
            except CaloraError as error:
                io.show_message(str(error))
            io.pause()


def main() -> None:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(threadName)s %(message)s")
    for sdk_logger in ("httpx", "httpx2", "httpcore", "httpcore2", "anthropic", "google.genai", "google_genai"):
        logging.getLogger(sdk_logger).setLevel(logging.WARNING)
    try:
        CliApp(AppFactory.build_controller()).run()
    except CaloraError as error:
        print(str(error))


if __name__ == "__main__":
    main()

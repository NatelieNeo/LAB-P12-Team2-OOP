"""Command-line entry point: python main.py"""
import logging
from typing import Optional

from dotenv import load_dotenv

from manager.controller import AppFactory, CaloraController
from manager.exceptions import CaloraError
from manager.io_manager import IOManager
from manager.job_manager import ProgressSpinner
from manager.models import UserProfile


class CliApp:
    """Terminal user interface. It owns no business logic: it asks the
    IOManager for input and output and the controller for the work."""

    def hello_world(self) -> None:
        print("Hello, world!")

    def __init__(self, controller: CaloraController, io: IOManager, spinner_factory=ProgressSpinner):
        self._controller = controller
        self._io = io
        self._spinner_factory = spinner_factory

    def run(self) -> None:
        io = self._io
        io.welcome()
        username = io.prompt_username()
        profile = self._controller.get_profile(username)
        if profile is not None:
            io.show_message(f"\nWelcome back, {profile.name}! Let's continue with your meal plan.")
        else:
            io.show_message(f"\nNo existing profile found for username: {username}. "
                            "Lets create a new profile for you.")
            profile = io.prompt_profile(username)
            self._controller.register_profile(profile)
            io.show_message("Profile setup complete!")

        handlers = {"1": self._create_plan, "2": self._view_history,
                    "3": self._view_profile, "4": self._update_profile}
        while True:
            choice = io.prompt_menu_choice()
            if choice == "5":
                io.show_message("Exiting the program. Goodbye!")
                return
            handler = handlers.get(choice)
            if handler is None:
                io.show_message("Please choose a number from 1 to 5.")
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

    def _view_profile(self, profile: UserProfile) -> UserProfile:
        self._io.show_profile(profile)
        self._io.pause()
        return profile

    def _update_profile(self, profile: UserProfile) -> UserProfile:
        changes = self._io.prompt_profile_changes(profile)
        updated = profile.updated_with(changes)
        self._controller.update_profile(profile.username, updated)
        self._io.pause()
        return updated

    @staticmethod
    def main(controller: Optional[CaloraController] = None) -> None:
        load_dotenv()
        logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")
        CliApp(controller or AppFactory.build_controller(), IOManager()).run()


if __name__ == "__main__":
    CliApp.main()

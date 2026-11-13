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

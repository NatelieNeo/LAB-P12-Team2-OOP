# Connects the IO layer to the AI layer
import json
from manager.data_manager import *
from manager.io_manager import *
from manager.ai_manager import call_ai
from manager.logic_manager import build_meal_plan, format_meal_plan


def main():
    # ----- IO LAYER -----
    input_record = {}
    welcome()

    # Check if the user exists
    username = get_username()
    if verify_user_profile("username", username):  #check if user profile exists
        # load user profile from JSON file
        users = load_user_profiles("username", username)

        name = users[list(users.keys())[0]].get("name", "User")
        print(f"\nWelcome back, {name}! Let's continue with your meal plan.")

        input_record = list(users.values())[0]  # Get the first (and only) user profile
    else:
        print(f"\nNo existing profile found for username: {username}. Lets create a new profile for you.")
        
        name = get_user_name()
        age = get_user_age()
        gender = get_user_gender()
        weight_kg = get_user_weight()
        height_cm = get_user_height()
        activity_level = get_activity_level()
        user_goal = get_user_goal()
        dietary_restrictions = get_dietary_restrictions()

        user_profile = [name, age, gender, weight_kg, height_cm, activity_level, user_goal, dietary_restrictions]  #store user profile in a list
        print("Profile setup complete! Here is your profile information: ", user_profile)

        # save user profile to JSON file
        user_data = {
            "username": username,
            "name": name,
            "age": age,
            "gender": gender,
            "weight_kg": weight_kg,
            "height_cm": height_cm,
            "activity_level": activity_level,
            "goal": user_goal,
            "dietary_restrictions": dietary_restrictions
        }

        save_user_profile(user_data)
        input_record = user_data 
    
    while True:
        user_option = option_list()  #get user option
        if user_option == "1":
            input_record = create_meal_plan(input_record, name)
            meal_plan_option(input_record, name)  #create meal plan

        elif user_option == "2":
            meal_plans = get_meal_plans_by_user(input_record["username"])

            if not meal_plans:
                print("No meal plans found for this user.")
                input("\nPress Enter to return to the main menu...")  # Pause before returning to the main menu
                continue

            for entry_id, meal_plan in meal_plans.items():
                print(f"\nMeal Plan ID: {entry_id}\n{format_meal_plan(meal_plan)}")
            input("Press Enter to return to the main menu...")  # Pause before returning to the main menu

        elif user_option == "3":
            user_data = load_user_profiles("username", username)
            print(f"\nHere is your profile information:\n{format_user_data(user_data)}")
            input("\nPress Enter to return to the main menu...")  # Pause before returning to the main menu

        elif user_option == "4":
            # Add code to update the user's profile here
            updated_data = {}
            print("\nUpdate your profile information. Leave blank to keep current value.")
            updated_data["name"] = get_user_name() or input_record["name"]
            updated_data["age"] = get_user_age(allow_null=True) or input_record["age"]
            updated_data["gender"] = get_user_gender(allow_null=True) or input_record["gender"]
            updated_data["weight_kg"] = get_user_weight(allow_null=True) or input_record["weight_kg"]
            updated_data["height_cm"] = get_user_height(allow_null=True) or input_record["height_cm"]
            updated_data["activity_level"] = get_activity_level(allow_null=True) or input_record["activity_level"]
            updated_data["goal"] = get_user_goal(allow_null=True) or input_record["goal"]
            updated_data["dietary_restrictions"] = get_dietary_restrictions(allow_null=True) or input_record["dietary_restrictions"]

            update_user_profile(input_record["username"], updated_data)
            input_record.update(updated_data)
            input("\nPress Enter to return to the main menu...")  # Pause before returning to the main menu

        elif user_option == "5":
            print("Exiting the program. Goodbye!")
            break


def meal_plan_option(input_record, name):
    # ----- AI LAYER -----
    ai_output = call_ai(input_record) # Send input_record to Gemini through ai_manager

    # Stop the program if the AI request failed
    if ai_output is None:
        print("Unable to generate meal recommendations.")
        return
    
    recommendations = ai_output.get("recommendations", [])

    # ----- LOGIC LAYER -----
    plan = build_meal_plan(recommendations, input_record) # Build a meal plan based on the AI recommendations and user input
    
    # read meal plan for testing
    # with open("meal_plan.json", "r") as f:
    #     plan = json.load(f)

    if plan is None:
        print("Unable to build a meal plan from the AI recommendations.")
        input("\nPress Enter to return to the main menu...")  # Pause before returning to the main menu

        return

    save_meal_plan(plan, input_record["username"])  # Save the meal plan to a JSON file

    print(format_meal_plan(plan))
    input("\nPress Enter to return to the main menu...")  # Pause before returning to the main menu

# Run the program
if __name__ == "__main__":
    main()
    
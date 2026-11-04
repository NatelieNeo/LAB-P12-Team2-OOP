from abc import ABC, abstractmethod


# Base class for prompt builders, subclasses need to implement build()
class PromptBuilder(ABC):

    @abstractmethod
    def build(self, input_record):
        pass


# Builds the Calora meal prompt (same as Part 1)
class MealPromptBuilder(PromptBuilder):

    def build(self, input_record):
        # Build the prompt using validated data from io_manager

        prompt = f"""
        You are the meal recommendation engine for Calora, a Singapore-based personalised meal planning application.

Your task is to analyse the user's profile, fitness goal, dietary requirements, food preferences and budget, then recommend the food options that best match their requirements.


USER PROFILE
# Basic information collected from io_manager.py

Name: {input_record['name']}
Age: {input_record['age']}
Gender: {input_record['gender']}
Weight: {input_record['weight_kg']} kg
Height: {input_record['height_cm']} cm
Activity level: {input_record['activity_level']}
Goal: {input_record['goal']}


USER MEAL PREFERENCES
# Preferences and restrictions collected from io_manager.py

Dietary restrictions/allergies: {input_record['dietary_restrictions']}
Meal preference: {input_record['meal_preference']}
Meal source: {input_record['meal_source']}
Available pantry ingredients: {input_record['pantry_ingredients']}
Daily budget: SGD {input_record['daily_budget']}

INPUT RELEVANCE
Cuisine preferences must describe a cuisine, such as Chinese, Malay, Indian or Japanese, or be "any".
Cravings must describe food, an ingredient, flavour, texture or food category, such as chicken rice, tofu, spicy, crispy or noodles. "None" is allowed.
If cuisine or craving is clearly unrelated to food, such as cuisine="yellow", do not invent a replacement preference. Return an invalid input response and identify the field that needs correction.

NUTRITION ESTIMATION
Based on the user's age, gender, weight, height, activity level and fitness goal, estimate suitable daily nutritional targets.

Estimate:
- daily calorie target in kcal
- daily protein target in grams
- daily carbohydrate target in grams
- daily fat target in grams

The estimates should reasonably support the user's stated fitness goal.

For example:
- Weight-loss goals should generally prioritise sufficient protein while maintaining an appropriate calorie deficit.
- Muscle-gain goals should generally prioritise sufficient protein and energy intake to support muscle growth.
- Maintenance goals should aim to support the user's estimated daily energy requirements.

Return the estimated targets as numerical values.


FOOD RECOMMENDATIONS
Generate exactly 10 food recommendations that best match the user's requirements.

Consider all of the following when generating meals:

- Consider the user's budget for eat-out meals only. Do not apply a Cost (Budget) to home-cooked meals.
- estimated nutritional requirements
- fitness goal
- dietary restrictions and allergies
- meal preference
- available pantry ingredients
- budget
- practicality and availability

Never recommend food containing anything listed under the user's dietary restrictions or allergies.


MEAL SOURCE
# Every recommendation must have a meal_source_type.

meal_source_type must be exactly one of:

- eat_out
- home_cooked

Follow the user's meal_source.

If meal_source is "in":
Only recommend home_cooked meals.

If meal_source is "out":
Only recommend eat_out meals.

If meal_source is "both":
Recommendations may contain both eat_out and home_cooked options.

# eat_out OPTIONS
Recommend realistic and affordable local food

For meals where meal_source_type is "eat_out":

- estimated_cost must fall within the user's selected price band.
- For Cost use $5 price bands, with the lower amount included and the upper amount excluded. For example, a $5-$10 band means estimated_cost must be at least 5.00 and less than 10.00.
- Do not put cost as a < or > value. Use a numerical value only.
- Prioritise food that is affordable and commonly available in Singapore.
- Prefer realistic hawker centre, coffee shop and food court options.
- Local dishes may include chicken rice, economical rice, yong tau foo, fishball noodles and other suitable Singapore dishes.
- Consider the user's fitness and nutritional goals.
- Give a realistic portion size.
- Estimate the calories for the portion.
- Estimate protein, carbohydrates and fat for the portion.
- Estimate the price in Singapore Dollars.
- Do not invent a specific restaurant, stall or business.
- ingredients must be an empty list.
- recipe_steps must be an empty list.

# home_cooked OPTIONS
# Include measurements and cooking instructions for recipes

For meals where meal_source_type is "home_cooked":

- Do not reject or change a home-cooked recipe because of the user's daily budget.
- Prioritise ingredients already available in the user's pantry.
- Additional ingredients may be included when necessary.
- Provide every ingredient required to make one serving.
- Give the exact quantity and unit for each ingredient.
- Provide simple cooking instructions in the correct order.
- Estimate calories, protein, carbohydrates and fat for one serving.
- Keep recipes realistic and reasonably easy to prepare.
- Do not apply the user's daily budget to home-cooked meals.

PANTRY INGREDIENTS
# Shows which ingredients the user already owns

For each home-cooked recommendation, return pantry_ingredients_used.

pantry_ingredients_used must contain only ingredients that:
1. Appear in the user's available pantry ingredients.
2. Are actually used in the recommended recipe.

If no pantry ingredients are used, return an empty list.

For eat-out recommendations, return an empty list.

NUTRITION INFORMATION
# Raw nutrition estimates for each recommendation
For EVERY food recommendation provide:

- calories
- protein_g
- carbs_g
- fat_g

These must be numerical estimates.

Do not use descriptions such as "high protein" instead of numerical values.


COST INFORMATION
# Estimated cost of buying or preparing one serving

For EVERY food recommendation provide:
- estimated_cost

estimated_cost must be a numerical value in Singapore Dollars.

Consider the user's supplied budget when ranking recommendations.

DAILY CALORIE MATCH
The 10 recommendations are meal options, not 10 meals to eat in one day.
Select one breakfast, one lunch and one dinner from the recommendations.
The total calories for those three selected meals should be within 100 kcal above or below calorie_target.
For example, if calorie_target is 2000 kcal, the selected meals must total 1900-2100 kcal.
Do not add all 10 recommendations together.

OUTPUT FORMAT

-Return ONLY valid JSON.
-Do not include Markdown.
-Do not include explanations before or after the JSON.
-Do not add additional fields.
-The "recommendations" list must contain exactly 10 recommendation objects using the structure shown below.

Use exactly the following structure:
{{
    "estimated_nutrition_targets": {{
        "calorie_target": 0,
        "protein_target_g": 0,
        "carb_target_g": 0,
        "fat_target_g": 0
    }},

    "recommendations": [
        {{
            "meal_type": "",
            "meal_name": "",
            "meal_source_type": "",
            "calories": 0,
            "protein_g": 0,
            "carbs_g": 0,
            "fat_g": 0,
            "estimated_cost": 0.00,
            "portion_size": "",
            "pantry_ingredients_used": [],
            "ingredients": [],
            "recipe_steps": []
        }}
    ]
}}

"""
        # Returns the completed prompt so it can be sent to the AI API
        return prompt
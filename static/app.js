const planForm = document.querySelector('form[action="/plan"]');
const loader = document.getElementById("plan-loader");

if (planForm && loader) {
    planForm.addEventListener("submit", () => {
        const submitButton = planForm.querySelector('button[type="submit"]');
        if (submitButton) {
            submitButton.disabled = true;
            submitButton.textContent = "Generating...";
        }
        loader.classList.add("is-visible");
    });
}

const mealSource = document.getElementById("meal_source");
const ingredients = document.getElementById("ingredients");

function toggleIngredients() {
    if (!mealSource || !ingredients) return;
    const eatingOut = mealSource.value === "out";
    ingredients.disabled = eatingOut;
    ingredients.placeholder = eatingOut
        ? "Not needed when eating out"
        : "e.g. chicken, rice, lettuce";
}

if (mealSource && ingredients) {
    mealSource.addEventListener("change", toggleIngredients);
    toggleIngredients();
}

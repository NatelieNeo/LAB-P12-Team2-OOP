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

// Grey out the ingredients box when the user is only eating out.
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
    toggleIngredients(); // also on page load, so a saved "out" profile starts greyed out
}

// Poll the background job until the meal plan is ready.
const planJob = document.getElementById("plan-job");

if (planJob && loader) {
    loader.classList.add("is-visible");
    const jobMessage = document.getElementById("plan-job-message");

    const pollJob = () => {
        fetch(planJob.dataset.statusUrl)
            .then((response) => response.json())
            .then((job) => {
                if (job.status === "DONE") {
                    window.location.href = planJob.dataset.resultUrl;
                } else if (job.status === "FAILED" || job.status === "NOT_FOUND") {
                    loader.classList.remove("is-visible");
                    jobMessage.textContent = job.message || "Unable to generate a meal plan right now.";
                    jobMessage.hidden = false;
                } else {
                    setTimeout(pollJob, 1000);
                }
            })
            .catch(() => setTimeout(pollJob, 2000));
    };
    pollJob();
}


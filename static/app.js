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

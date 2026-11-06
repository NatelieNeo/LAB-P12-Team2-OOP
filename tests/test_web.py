"""WebApp through Flask's test client, using the offline stub provider."""
import time

import pytest

from app import WebApp
from tests.conftest import make_profile

PROFILE_FORM = {"username": "sam", "name": "Sam", "age": "30", "gender": "F", "weight_kg": "60",
                "height_cm": "165", "activity_level": "sedentary", "goal": "maintain",
                "dietary_restrictions": "nil"}
PLAN_FORM = {"meal_preference": "rice dishes", "meal_source": "in", "ingredients": "eggs, rice", "budget": "0"}


@pytest.fixture
def web(controller_factory, response_text):
    controller, provider = controller_factory(response_text)
    client = WebApp(controller).flask_app.test_client()
    return client, controller, provider


def sign_up(client):
    client.post("/login", data={"username": "sam"})
    return client.post("/profile/create", data=PROFILE_FORM)


def wait_for_plan(client, job_id, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = client.get(f"/plan/status/{job_id}").get_json()["status"]
        if status in ("DONE", "FAILED"):
            return status
        time.sleep(0.01)
    raise AssertionError("job did not finish")


def start_plan(client, form=None):
    page = client.post("/plan", data=form or PLAN_FORM)
    text = page.get_data(as_text=True)
    assert page.status_code == 200 and "plan-job" in text
    return text.split("/plan/status/")[1].split('"')[0]


def test_welcome_page_for_signed_out_visitors(web):
    client, *_ = web
    assert "username" in client.get("/").get_data(as_text=True)


def test_new_user_registration_flow(web):
    client, controller, _ = web
    page = client.post("/login", data={"username": "sam"})
    assert "Create profile" in page.get_data(as_text=True)
    response = sign_up(client)
    assert response.status_code == 302 and response.headers["Location"].endswith("/profile")
    assert controller.get_profile("sam").goal == "maintain"
    assert "Sam" in client.get("/profile").get_data(as_text=True)


def test_blank_username_is_rejected(web):
    client, *_ = web
    assert client.post("/login", data={"username": "  "}).status_code == 400


def test_invalid_profile_form_is_redisplayed_with_a_message(web):
    client, controller, _ = web
    client.post("/login", data={"username": "sam"})
    response = client.post("/profile/create", data={**PROFILE_FORM, "age": "5"})
    assert response.status_code == 400 and "age between" in response.get_data(as_text=True)
    assert controller.get_profile("sam") is None


def test_duplicate_username_is_refused(web):
    client, *_ = web
    sign_up(client)
    other = client.application.test_client()
    page = other.post("/profile/create", data=PROFILE_FORM)
    assert "already exists" in page.get_data(as_text=True)


def test_background_plan_generation_end_to_end(web):
    client, controller, provider = web
    sign_up(client)
    job_id = start_plan(client)
    assert wait_for_plan(client, job_id) == "DONE"
    result = client.get(f"/plan/result/{job_id}").get_data(as_text=True)
    assert "Garlic Chicken Pasta" in result and "Shopping list" in result and "2 slices wholemeal bread" in result
    assert len(controller.get_history("sam")) == 1
    assert provider.call_count == 1


def test_allergen_notice_hidden_for_home_cooked_and_shown_for_eat_out(web):
    client, *_ = web
    sign_up(client)
    home = start_plan(client, {**PLAN_FORM, "meal_source": "in"})
    wait_for_plan(client, home)
    assert "cannot guarantee" not in client.get(f"/plan/result/{home}").get_data(as_text=True)
    out = start_plan(client, {**PLAN_FORM, "meal_source": "out"})
    wait_for_plan(client, out)
    assert "cannot guarantee" in client.get(f"/plan/result/{out}").get_data(as_text=True)


def test_failed_job_reports_a_friendly_message(controller_factory):
    controller, _ = controller_factory("not json")
    client = WebApp(controller).flask_app.test_client()
    sign_up(client)
    job_id = start_plan(client)
    assert wait_for_plan(client, job_id) == "FAILED"
    status = client.get(f"/plan/status/{job_id}").get_json()
    assert "unusable" in status["message"]
    result = client.get(f"/plan/result/{job_id}")
    assert result.status_code == 503 and "unusable" in result.get_data(as_text=True)


def test_invalid_plan_form_keeps_the_user_on_the_form(web):
    client, *_ = web
    sign_up(client)
    response = client.post("/plan", data={**PLAN_FORM, "meal_source": "teleport"})
    assert response.status_code == 400 and "in, out or both" in response.get_data(as_text=True)


def test_other_users_cannot_read_a_job(web):
    client, *_ = web
    sign_up(client)
    job_id = start_plan(client)
    wait_for_plan(client, job_id)
    stranger = client.application.test_client()
    assert stranger.get(f"/plan/status/{job_id}").status_code == 404
    assert stranger.get(f"/plan/result/{job_id}").status_code == 302


def test_unknown_job_is_a_404_for_status(web):
    client, *_ = web
    sign_up(client)
    assert client.get("/plan/status/nope").status_code == 404


def test_profile_edit_and_update(web):
    client, controller, _ = web
    sign_up(client)
    assert "Save profile changes" in client.get("/profile/edit").get_data(as_text=True)
    page = client.post("/profile/update", data={**PROFILE_FORM, "age": "31", "username": "ignored"})
    assert "Profile updated successfully." in page.get_data(as_text=True)
    assert controller.get_profile("sam").age == 31 and controller.get_profile("ignored") is None
    bad = client.post("/profile/update", data={**PROFILE_FORM, "weight_kg": "abc"})
    assert bad.status_code == 400 and controller.get_profile("sam").weight_kg == 60.0


def test_update_keeps_the_saved_preferences(web):
    client, controller, _ = web
    sign_up(client)
    wait_for_plan(client, start_plan(client))
    client.post("/profile/update", data={**PROFILE_FORM, "age": "31"})
    assert controller.get_profile("sam").preferences["meal_preference"] == "rice dishes"


def test_history_page_lists_saved_plans(web):
    client, *_ = web
    sign_up(client)
    assert "No meal plans found" in client.get("/plans").get_data(as_text=True)
    wait_for_plan(client, start_plan(client))
    assert "Meal plan 1" in client.get("/plans").get_data(as_text=True)


def test_logout_clears_the_session(web):
    client, *_ = web
    sign_up(client)
    assert client.post("/logout").status_code == 302
    assert client.get("/profile").status_code == 302


@pytest.mark.parametrize("path", ["/profile", "/profile/edit", "/plans"])
def test_signed_out_users_are_redirected(web, path):
    client, *_ = web
    assert client.get(path).status_code == 302


def test_saved_preferences_prefill_the_form(web):
    client, *_ = web
    sign_up(client)
    wait_for_plan(client, start_plan(client))
    page = client.get("/").get_data(as_text=True)
    assert 'value="rice dishes"' in page and 'value="eggs, rice"' in page


def test_module_level_app_is_importable_for_gunicorn():
    import app
    assert app.app.name and "plan_status" in app.app.view_functions


def test_returning_user_signs_in_to_profile(web):
    client, controller, _ = web
    controller.register_profile(make_profile(username="tim", name="Tim"))
    response = client.post("/login", data={"username": "tim"})
    assert response.status_code == 302

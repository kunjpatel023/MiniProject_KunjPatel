import streamlit as st
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time
import os

st.set_page_config(page_title="Mood Playlist Generator", layout="centered")

st.title("🎵 Mood-Based Playlist Generator")
st.markdown("""
Welcome to the **Mood Playlist Generator**! Select a mood below and this app will automate searching for a matching playlist on YouTube using Chrome.
""")

st.subheader("Choose Your Mood")
mood_options = {
    "Happy": "happy songs playlist",
    "Calm": "calm relaxing music",
    "Energetic": "energetic workout music"
}
selected_mood = st.selectbox("Pick a mood:", list(mood_options.keys()))


def automate_playlist_search(mood_query):
    try:
        driver = webdriver.Chrome()
        driver.get("https://www.youtube.com/")
        time.sleep(3) 


        search_bar = driver.find_element(By.NAME, "search_query")
        search_bar.send_keys(mood_query)
        search_bar.send_keys(Keys.ENTER)
        time.sleep(3) 

        # Click the first video/playlist which show in youtube
        first_result = driver.find_element(By.CSS_SELECTOR, "ytd-video-renderer a#thumbnail")
        first_result.click()
        time.sleep(3) 

        log_message = f"[{time.ctime()}] Searched and played '{mood_query}' on YouTube"
        with open("automation_log.txt", "a") as log_file:
            log_file.write(log_message + "\n")

        return driver, f"Automation successful! Playing '{mood_query}' on YouTube. Click 'Close Browser' when ready."
    except Exception as e:
        return None, f"Automation failed: {str(e)}"

st.subheader("Generate Your Playlist")
st.markdown("*Click to automate the playlist search on YouTube.*")
automate_button = st.button("Run Automation")

if 'driver' not in st.session_state:
    st.session_state.driver = None

if automate_button:
    with st.spinner("Running automation..."):
        driver, result = automate_playlist_search(mood_options[selected_mood])
        st.session_state.driver = driver
        if "successful" in result:
            st.success(result)
        else:
            st.error(result)
            if driver:
                driver.quit()
                st.session_state.driver = None

st.subheader("Finish Automation")
st.markdown("*Click to close the browser*")
close_button = st.button("Close Browser")

if close_button and st.session_state.driver:
    st.session_state.driver.quit()
    st.session_state.driver = None
    st.success("Browser closed successfully!")


if os.path.exists("automation_log.txt"):
    st.subheader("Automation Log")
    with open("automation_log.txt", "r") as log_file:
        st.text(log_file.read())


st.markdown("""
<style>
    .stButton>button {
        background-color: #FF0000;  /* YouTube red */
        color: white;
        border-radius: 5px;
    }
    .stSelectbox {
        border: 1px solid #ddd;
        border-radius: 5px;
    }
    .stSpinner {
        color: #FF0000;
    }
</style>
""", unsafe_allow_html=True)
# auth_gate.py
import os, streamlit as st

def check_password():
    """Simple password gate using an environment variable ASHA_APP_PASSWORD."""
    def password_entered():
        if st.session_state["password"] == os.environ.get("ASHA_APP_PASSWORD", ""):
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if st.session_state.get("password_correct", False):
        return True

    st.write("### 🔒 Private Preview")
    st.text_input("Enter access password", type="password", key="password", on_change=password_entered)
    if "password_correct" in st.session_state and not st.session_state["password_correct"]:
        st.error("Password incorrect.")
    st.stop()

import streamlit as st

st.set_page_config(
    page_title="AI Biomechanics",
    layout="centered",
)

st.title("AI Biomechanics")
st.write("Choose your analysis:")

c1, c2 = st.columns(2)

with c1:
    st.markdown("### ⚾ Hitting")
    if st.button("Open Hitting Biomechanics", use_container_width=True):
        st.switch_page("app.py")

with c2:
    st.markdown("### 🧢 Pitching")
    if st.button("Open Pitching Biomechanics", use_container_width=True):
        st.switch_page("pitching/app.py")

import streamlit as st
import requests
import time

st.set_page_config(layout="wide")

st.title("🧠 FIE — Live Intelligence")

placeholder = st.empty()

while True:
    data = requests.get("http://localhost:8000/live").json()

    with placeholder.container():
        st.subheader("🔥 Top Prediction")
        st.write(f"{data['event']}")
        probability = data['prediction']['probability']
        st.write(f"Probability: {probability * 100:.1f}%")
        st.progress(probability)

        st.divider()

        st.subheader("💰 Market Edge")
        if data["market"]:
            edge = data["market"]["edge"]

            if edge > 0.15:
                st.warning("⚡ HIGH EDGE OPPORTUNITY")

            st.write(f"Market: {data['market']['market_probability']*100:.1f}%")
            st.write(f"Edge: {edge}")
            st.write(f"Signal: {data['market']['signal']}")

            if edge > 0:
                st.success("BUY YES")
            else:
                st.error("BUY NO")

        st.divider()

        st.subheader("🧠 Agent Debate")
        for agent in data["agents"]:
            st.write(f"{agent['agent']} → {agent['probability']:.2f}")
            st.caption(agent["reasoning"])

    time.sleep(10)

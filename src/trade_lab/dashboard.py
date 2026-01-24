"""Main Streamlit dashboard application."""

import streamlit as st


def main():
    """Run the Streamlit dashboard application."""
    st.set_page_config(
        page_title="Options Dashboard",
        page_icon="📊",
        layout="wide",
    )

    st.title("📊 Options Trading Dashboard")
    st.write("Welcome to your options trading analytics dashboard!")

    st.info("This is a starter template. Add your matplotlib charts and data analysis here.")


if __name__ == "__main__":
    main()

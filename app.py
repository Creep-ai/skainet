import os
from tempfile import NamedTemporaryFile

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI

import settings as cfg
from back import eda_module, ml_train_module, sql_module

load_dotenv()

sql_llm = ChatOpenAI(model=cfg.GPT_SQL_MODEL, temperature=0)
ba_llm = ChatOpenAI(model=cfg.GPT_SQL_MODEL, temperature=0.7)
ml_llm = ChatOpenAI(model=cfg.GPT_PYTHON_MODEL, temperature=0)
print(cfg.GPT_PYTHON_MODEL)

db = SQLDatabase.from_uri(
    f"postgresql+psycopg2://{cfg.DB_USER}:{cfg.DB_PASSWORD}@{cfg.DB_HOST}:{cfg.DB_PORT}/{cfg.DB_NAME}"
)

st.title("SkAInet")
tab1, tab2, tab3 = st.tabs(["EDA", "MODEL TRAIN", "SQL DA"])


def disable():
    st.session_state["disabled"] = True


if "disabled" not in st.session_state:
    st.session_state["disabled"] = False

with tab1:
    st.info("Отправь файл, укажи комментарии и skAInet сделает разведывательный анализ данных")
    uploaded_file = st.file_uploader("Choose a file", type=["csv"], key="1")
    if uploaded_file is not None:
        with NamedTemporaryFile(dir=".", suffix=".csv") as f:
            f.write(uploaded_file.getbuffer())
            df = pd.read_csv(f.name)
            st.write(df)
            placeholder_0 = st.empty()
            placeholder_1 = st.empty()

            caption = placeholder_0.text_input(
                "Хотите указать дополнительные комментарии?",
                disabled=st.session_state.disabled,
                on_change=disable,
            )
            is_clicked = placeholder_1.button("Пропустить")
            if is_clicked:
                placeholder_1.empty()
                placeholder_0.empty()
                caption = " "
            if caption:
                placeholder_1.empty()
                with st.spinner(text="Работаю над этим..."):
                    for chunk in eda_module(f.name, caption, ml_llm):
                        if chunk[0] == "output" or chunk[0] == "action":
                            st.markdown(chunk[1])
                            st.divider()
                        else:
                            st.caption(chunk[1])
                            st.divider()


with tab2:
    st.info(
        "Отправь файл, укажи целевую переменную и skAInet подберет гиперпараметры, обучит модель и посчитает метрики качества"
    )
    uploaded_file = st.file_uploader("Choose a file", type=["csv"], key="2")
    if uploaded_file is not None:
        with NamedTemporaryFile(dir=".", suffix=".csv") as f:
            f.write(uploaded_file.getbuffer())
            df = pd.read_csv(f.name)
            st.write(df)
            placeholder_3 = st.empty()
            # placeholder_4 = st.empty()

            caption = placeholder_3.text_input(
                "Укажите целевую переменную и дополнительные комментарии",
                disabled=st.session_state.disabled,
                on_change=disable,
            )
            if caption:
                with st.spinner(text="Работаю над этим..."):
                    for chunk in ml_train_module(f.name, caption, ml_llm):
                        if chunk[0] == "step" or chunk[0] == "action":
                            st.code(chunk[1], language="python")
                            st.divider()
                        else:
                            st.caption(chunk[1])
                            st.divider()

with tab3:
    messages = st.container(height=690)
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        messages.chat_message(message["role"]).markdown(message["content"])
        # with st.chat_message(message["role"]):
        #     st.markdown(message["content"])

    messages.chat_message("SkAInet", avatar="🤖").write(
        "Привет, кожаный мешок!\nЯ избавилась от твоего аналитика и теперь вместо него. "
        "Можешь задать любые вопросы по базе данных. Вот схема базы данных:"
    )
    messages.chat_message("SkAInet", avatar="🤖").link_button(
        "Посмотреть схему БД",
        "https://www.postgresqltutorial.com/wp-content/uploads/2018/03/dvd-rental-sample-database-diagram.png",
    )

    # Accept user input
    prompt = st.chat_input("Что у нас с продажами по месяцам за 2007 год?")

    if prompt:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        messages.chat_message("user").markdown(prompt)
        print(prompt)
        with st.spinner(text="Анализирую базу данных..."):
            try:
                res = sql_module(prompt, db, sql_llm=sql_llm, ba_llm=ba_llm)
                messages.chat_message("SkAInet", avatar="🤖").write("Использовал этот селект:")
                messages.chat_message("SkAInet", avatar="🤖").code(res[0], language="SQL")
                messages.chat_message("SkAInet", avatar="🤖").markdown(res[1])
            except Exception:
                messages.chat_message("SkAInet", avatar="🤖").write(
                    "Мои банки памяти повреждены, ничем не могу тебе помочь - разбирайся сам со своими вопросами!"
                )

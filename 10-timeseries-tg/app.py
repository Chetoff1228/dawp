import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats.mstats import winsorize

# Настройка страницы
st.set_page_config(page_title="Telegram Chat Dashboard", layout="wide")

st.title("📊 Аналитика Telegram-чата")

# --- Sidebar: ввод пути к JSON-файлу ---
st.sidebar.header("Настройки")
json_path = st.sidebar.text_input(
    "Путь к JSON-файлу",
    value="result.json",
)


# --- Функция загрузки данных ---
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        st.error(f"Не удалось загрузить файл: {e}")
        return pd.DataFrame()

    messages = [msg for msg in data.get("messages", []) if msg.get("type") == "message"]
    df = pd.DataFrame(messages)
    if df.empty:
        return df

    # Преобразуем колонки
    df["date"] = pd.to_datetime(df["date"])
    df["from"] = df["from"].astype(str)
    # Приводим текст к строке, чтобы избежать смешения типов
    df["text"] = df["text"].apply(lambda x: x if isinstance(x, str) else "")
    # Подсчёт реакций
    df["reactions_count"] = df.get("reactions", pd.Series()).apply(
        lambda r: sum(item.get("count", 1) for item in r) if isinstance(r, list) else 0
    )

    # Установим индекс
    df.set_index("date", inplace=True)
    return df


# Загружаем данные
df = load_data(json_path)
if df.empty:
    st.stop()

# --- Подготовка признаков ---
df["day"] = df.index.date
df["hour"] = df.index.hour
df["weekday"] = df.index.day_name()
df["month"] = df.index.to_period("M").astype(str)

# --- Агрегации ---
msgs_per_min = df.resample("1min").size()
msgs_per_hour = df.resample("H").size()
msgs_per_day = df.groupby("day").size()
weekday_counts = (
    df.groupby("weekday")
    .size()
    .reindex(
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    )
)
month_counts = df.groupby("month").size()

# --- Основные метрики ---
st.header("Основные показатели")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(label="Всего сообщений", value=len(df))
with col2:
    st.metric(label="Уникальных участников", value=df["from"].nunique())
with col3:
    top_day = msgs_per_day.idxmax()
    top_day_count = msgs_per_day.max()
    st.metric(label="Самый активный день", value=f"{top_day} ({top_day_count})")
with col4:
    top_user = df["from"].value_counts().idxmax()
    top_user_count = df["from"].value_counts().max()
    st.metric(
        label="Самый активный пользователь", value=f"{top_user} ({top_user_count})"
    )

# --- Временные ряды ---
st.header("Временная активность")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

msgs_per_hour.plot(ax=axes[0])
axes[0].set_title("Сообщения по часам")
axes[0].set_xlabel("Время")
axes[0].set_ylabel("Число сообщений")

msgs_per_day.plot(ax=axes[1])
axes[1].set_title("Сообщения по дням")
axes[1].set_xlabel("Дата")
axes[1].set_ylabel("Число сообщений")

st.pyplot(fig)

# --- Распределение по дням недели и месяцам ---
st.header("Распределение по дням недели и месяцам")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

weekday_counts.plot(kind="bar", ax=axes[0])
axes[0].set_title("По дням недели")
axes[0].set_ylabel("Количество сообщений")

month_counts.plot(kind="bar", ax=axes[1])
axes[1].set_title("По месяцам")
axes[1].set_ylabel("Количество сообщений")

st.pyplot(fig)

# --- Топ-10 пользователей ---
st.header("Топ-10 активных пользователей")
top_users = df["from"].value_counts().head(10).reset_index()
top_users.columns = ["Пользователь", "Сообщений"]
st.table(top_users)

# --- Топ-10 сообщений по реакциям ---
st.header("Топ-10 сообщений по числу реакций")
top_reacted = df.nlargest(10, "reactions_count")[
    ["from", "text", "reactions_count"]
].rename(
    columns={"from": "Пользователь", "text": "Текст", "reactions_count": "Реакций"}
)
st.table(top_reacted)

# --- Сезонность: день в году ---
st.header("Сезонность по дню года")

day_stats = msgs_per_day.reset_index(name="count")
day_stats["m_d"] = pd.to_datetime(day_stats["day"]).dt.strftime("%m-%d")
seasonal = day_stats.groupby("m_d")["count"].mean().reset_index()
seasonal["count_w"] = winsorize(seasonal["count"], limits=[0.1, 0.1])

fig, axes = plt.subplots(1, 2, figsize=(15, 4))

axes[0].plot(seasonal["m_d"], seasonal["count"])
axes[0].set_title("Среднее сообщений по дню года")
axes[0].tick_params(axis="x", rotation=90)

axes[1].plot(seasonal["m_d"], seasonal["count_w"])
axes[1].set_title("Винзоризированный тренд")
axes[1].tick_params(axis="x", rotation=90)

st.pyplot(fig)

st.markdown("---")
st.caption("© 2025 Mimika AI Analytics")

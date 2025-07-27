import streamlit as st
import pandas as pd
import numpy as np
import os

DATA_FILE = 'model_data.csv'

if not os.path.exists(DATA_FILE):
    pd.DataFrame(columns=['outcome', 'next_outcome']).to_csv(DATA_FILE, index=False)

df = pd.read_csv(DATA_FILE)

def update_model(sequence):
    records = []
    for i in range(len(sequence)-1):
        records.append({'outcome': sequence[i], 'next_outcome': sequence[i+1]})
    if records:
        new_df = pd.DataFrame(records)
        new_df.to_csv(DATA_FILE, mode='a', header=False, index=False)
        return new_df
    return pd.DataFrame()

def compute_transition_matrix(df):
    trans = pd.crosstab(df['outcome'], df['next_outcome'], normalize='index')
    for i in range(6):
        if i not in trans.index:
            trans.loc[i] = 0
    for j in range(6):
        if j not in trans.columns:
            trans[j] = 0
    return trans.sort_index().sort_index(axis=1)

def apply_bias_adjustments(trans, streak):
    bias_trans = trans.copy()
    last = streak['value']
    length = streak['length']
    if length >= 3 and last in bias_trans.index:
        reduce_amt = min(0.2, bias_trans.loc[last, last])
        bias_trans.loc[last, last] -= reduce_amt
        others = [i for i in range(6) if i != last]
        bias_trans.loc[last, others] += reduce_amt / len(others)
    return bias_trans

def get_last_streak(sequence):
    if not sequence:
        return {'value': None, 'length': 0}
    val = sequence[-1]
    length = 1
    for x in reversed(sequence[:-1]):
        if x == val:
            length += 1
        else:
            break
    return {'value': val, 'length': length}

st.title("Above/Below Predictor")

st.sidebar.header("Add Data / Predict")

input_mode = st.sidebar.radio("Input Mode", ["Manual Entry", "Upload CSV"])
sequence = []

if input_mode == "Manual Entry":
    s = st.sidebar.text_input("Enter outcomes (comma-separated 0-5)", "")
    try:
        sequence = [int(x) for x in s.split(',') if x.strip()!='']
    except:
        st.sidebar.error("Invalid input. Use numbers 0-5 separated by commas.")
elif input_mode == "Upload CSV":
    uploaded = st.sidebar.file_uploader("Upload CSV with 'outcome' column", type="csv")
    if uploaded:
        df_uploaded = pd.read_csv(uploaded)
        if 'outcome' in df_uploaded.columns:
            sequence = df_uploaded['outcome'].dropna().astype(int).tolist()
        else:
            st.sidebar.error("CSV must have 'outcome' column.")

if sequence:
    new_data = update_model(sequence)
    if not new_data.empty:
        df = pd.read_csv(DATA_FILE)
        st.sidebar.success(f"Added {len(new_data)} new transitions to model.")

    trans = compute_transition_matrix(df)
    streak = get_last_streak(sequence)
    biased_trans = apply_bias_adjustments(trans, streak)

    last = sequence[-1]
    probs = biased_trans.loc[last].fillna(0)

    prob_above = probs[3:].sum()
    prob_below = probs[:3].sum()
    pred_binary = "Above 2" if prob_above > prob_below else "Below 2"
    conf_binary = max(prob_above, prob_below)

    st.subheader("Prediction")
    st.write(f"**Above/Below 2:** {pred_binary}  (Confidence: {conf_binary:.1%})")

    st.subheader("Streak Info")
    st.write(f"Current streak: {streak['value']} repeated {streak['length']} times")

    st.subheader("Transition Matrix (after bias)")
    st.dataframe(biased_trans)

st.sidebar.write(f"Model transitions stored: {len(df)}")

import streamlit as st
import torch, torch.nn as nn
import numpy as np, pandas as pd

st.set_page_config(page_title="Dashboard Mochi", page_icon="🍡")

@st.cache_resource
def muat_model():
    ckpt = torch.load("model_mochi.pth", map_location="cpu", weights_only=True)
    net = nn.Sequential(nn.Linear(6, 16), nn.ReLU(), nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 3))
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    return net, ckpt

net, ckpt = muat_model()
VAR = ["Strawberry", "Matcha", "Red Velvet", "Mocca", "Coklat"]
SARAN = {
    "Pecinta mochi": "Tawarkan Strawberry dan paket langganan mingguan.",
    "Pembeli hadiah/acara": "Tawarkan paket box/hampers isi beragam varian.",
    "Pembeli hemat": "Tawarkan Coklat dan promo harga hemat.",
}

st.title("🍡 Dashboard Mochi")
st.caption("Prototipe berbasis data simulasi (Tugas Besar Machine Learning)")

tab1, tab2 = st.tabs(["Prediksi segmen pembeli", "Varian terlaris"])

with tab1:
    usia = st.number_input("Usia pembeli", min_value=5, max_value=80, value=10)
    pilihan = st.multiselect("Varian yang pernah dibeli", VAR)
    if st.button("Cek segmen"):
        x = [usia] + [1 if v in pilihan else 0 for v in VAR]
        x = (np.array(x) - np.array(ckpt["mean"])) / np.array(ckpt["scale"])
        with torch.no_grad():
            logit = net(torch.tensor(x, dtype=torch.float32).unsqueeze(0))
            p = torch.softmax(logit, dim=1)[0].numpy()
        i = int(p.argmax())
        segmen = ckpt["kelas"][i]
        st.success(f"Segmen: {segmen} (keyakinan {p[i]*100:.1f}%)")
        st.write(SARAN[segmen])
        st.bar_chart(pd.DataFrame({"Probabilitas": p}, index=ckpt["kelas"]))

with tab2:
    data = pd.read_csv("data_mochi.csv")
    st.bar_chart((data[VAR].mean() * 100).round(1).sort_values(ascending=False))
    st.info("Rekomendasi bundling: Matcha + Red Velvet (lift tertinggi 1,16 pada analisis Apriori).")

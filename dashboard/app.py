"""
Dashboard Streamlit para monitoramento do modelo
"""
import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import json
from datetime import datetime
from pathlib import Path

# Configuração da página
st.set_page_config(
    page_title="Passos Mágicos - Monitoramento Ponto de Virada",
    layout="wide"
)

# Configurações - usar variável de ambiente para deploy
API_URL = os.environ.get("API_URL", "http://localhost:8000")
LOGS_DIR = Path(__file__).parent.parent / "logs"


def get_api_metrics():
    """Busca métricas da API."""
    try:
        response = requests.get(f"{API_URL}/metrics", timeout=5)
        return response.json()
    except:
        return None


def get_model_info():
    """Busca informações do modelo."""
    try:
        response = requests.get(f"{API_URL}/model/info", timeout=5)
        return response.json()
    except:
        return None


def load_predictions_log():
    """Carrega log de predições."""
    log_path = LOGS_DIR / "predictions.jsonl"
    if not log_path.exists():
        return []
    
    predictions = []
    with open(log_path, "r") as f:
        for line in f:
            if line.strip():
                predictions.append(json.loads(line))
    return predictions


# Header
st.title("Passos Mágicos - Dashboard Ponto de Virada")
st.markdown("---")

# Status da API
col1, col2, col3, col4 = st.columns(4)

metrics = get_api_metrics()
model_info = get_model_info()

with col1:
    if metrics:
        st.metric("Status API", "Online")
    else:
        st.metric("Status API", "Offline")

with col2:
    if metrics:
        st.metric("Total de Requisições", metrics.get("total_requests", 0))
    else:
        st.metric("Total de Requisições", "N/A")

with col3:
    if metrics:
        st.metric("Total de Predições", metrics.get("total_predictions", 0))
    else:
        st.metric("Total de Predições", "N/A")

with col4:
    if metrics:
        st.metric("Uptime", f"{metrics.get('uptime_seconds', 0):.0f}s")
    else:
        st.metric("Uptime", "N/A")

st.markdown("---")

# Informações do Modelo
st.header("Informações do Modelo")

if model_info:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Detalhes")
        st.write(f"**Tipo:** {model_info.get('model_type', 'N/A')}")
        st.write(f"**Versão:** {model_info.get('version', 'N/A')}")
        st.write(f"**Treinado em:** {model_info.get('trained_at', 'N/A')}")
    
    with col2:
        st.subheader("Métricas de Treino")
        train_metrics = model_info.get("metrics", {})
        
        metrics_df = pd.DataFrame([
            {"Métrica": "Accuracy", "Valor": train_metrics.get("accuracy", 0)},
            {"Métrica": "Precision", "Valor": train_metrics.get("precision", 0)},
            {"Métrica": "Recall", "Valor": train_metrics.get("recall", 0)},
            {"Métrica": "F1-Score", "Valor": train_metrics.get("f1_score", 0)},
            {"Métrica": "ROC-AUC", "Valor": train_metrics.get("roc_auc", 0)},
        ])
        
        fig = px.bar(
            metrics_df, 
            x="Métrica", 
            y="Valor",
            color="Valor",
            color_continuous_scale="viridis"
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Modelo não carregado. Execute o treinamento primeiro.")

st.markdown("---")

# Monitoramento de Predições
st.header("Monitoramento de Predições")

predictions = load_predictions_log()

if predictions:
    # Converter para DataFrame
    df_preds = pd.DataFrame([
        {
            "timestamp": p["timestamp"],
            "prediction": p["output"]["prediction"],
            "confidence": p["output"]["confidence"],
            "label": p["output"].get("label", "Ponto de Virada Provável" if p["output"]["prediction"] == 1 else "Ponto de Virada Improvável")
        }
        for p in predictions
    ])
    
    df_preds["timestamp"] = pd.to_datetime(df_preds["timestamp"])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Distribuição de Predições")
        dist = df_preds["label"].value_counts()
        fig = px.pie(
            values=dist.values,
            names=dist.index,
            color_discrete_sequence=["#3498db", "#95a5a6"]
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Confiança das Predições")
        fig = px.histogram(
            df_preds, 
            x="confidence",
            nbins=20,
            color="label",
            color_discrete_map={"Ponto de Virada Provável": "#3498db", "Ponto de Virada Improvável": "#95a5a6"}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Predições ao longo do tempo
    st.subheader("Predições ao Longo do Tempo")
    
    df_preds["date"] = df_preds["timestamp"].dt.date
    daily_counts = df_preds.groupby(["date", "label"]).size().reset_index(name="count")
    
    fig = px.bar(
        daily_counts,
        x="date",
        y="count",
        color="label",
        color_discrete_map={"Ponto de Virada Provável": "#3498db", "Ponto de Virada Improvável": "#95a5a6"},
        barmode="stack"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Tabela de últimas predições
    st.subheader("Últimas Predições")
    st.dataframe(
        df_preds.tail(10).sort_values("timestamp", ascending=False),
        use_container_width=True
    )
else:
    st.info("Nenhuma predição registrada ainda.")

st.markdown("---")

# Monitoramento de Drift
st.header("Detecção de Drift")

if predictions and len(predictions) > 10:
    # Calcular drift de predições
    recent_predictions = [p["output"]["prediction"] for p in predictions[-100:]]
    class_1_ratio = sum(recent_predictions) / len(recent_predictions)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Proporção Ponto de Virada",
            f"{class_1_ratio:.1%}",
            delta=f"{(class_1_ratio - 0.5) * 100:.1f}%" if class_1_ratio != 0.5 else "0%"
        )
    
    with col2:
        drift_status = "!! Possível Drift !!" if abs(class_1_ratio - 0.5) > 0.2 else "Normal"
        st.metric("Status de Drift", drift_status)
    
    with col3:
        st.metric("Predições Analisadas", len(recent_predictions))
    
    # Gráfico de drift ao longo do tempo
    if len(predictions) > 50:
        st.subheader("Evolução da Distribuição de Predições")
        
        window_size = 50
        ratios = []
        
        for i in range(window_size, len(predictions)):
            window = predictions[i-window_size:i]
            preds = [p["output"]["prediction"] for p in window]
            ratios.append({
                "index": i,
                "class_1_ratio": sum(preds) / len(preds)
            })
        
        df_drift = pd.DataFrame(ratios)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_drift["index"],
            y=df_drift["class_1_ratio"],
            mode="lines",
            name="Proporção Classe 1"
        ))
        fig.add_hline(y=0.5, line_dash="dash", line_color="gray", annotation_text="Baseline (50%)")
        fig.add_hline(y=0.7, line_dash="dot", line_color="red", annotation_text="Threshold Superior")
        fig.add_hline(y=0.3, line_dash="dot", line_color="red", annotation_text="Threshold Inferior")
        
        fig.update_layout(
            xaxis_title="Predições",
            yaxis_title="Proporção de Ponto de Virada",
            yaxis_range=[0, 1]
        )
        
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Dados insuficientes para análise de drift. Faça mais predições.")

st.markdown("---")

# Métricas de Performance da API
st.header("Performance da API")

if metrics:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Latência")
        latency = metrics.get("latency", {})
        
        latency_df = pd.DataFrame([
            {"Métrica": "Média", "ms": latency.get("avg_ms", 0)},
            {"Métrica": "P95", "ms": latency.get("p95_ms", 0)},
            {"Métrica": "P99", "ms": latency.get("p99_ms", 0)},
            {"Métrica": "Máximo", "ms": latency.get("max_ms", 0)},
        ])
        
        fig = px.bar(latency_df, x="Métrica", y="ms", color="ms", color_continuous_scale="reds")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Requisições por Status")
        status_counts = metrics.get("requests_by_status", {})
        
        if status_counts:
            fig = px.pie(
                values=list(status_counts.values()),
                names=[f"HTTP {k}" for k in status_counts.keys()]
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Nenhuma requisição registrada.")
else:
    st.warning("API não disponível.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: gray;">
        Passos Mágicos - MLOps Dashboard | Desenvolvido para o Datathon FIAP | Thiago Pagotto RM361741
    </div>
    """,
    unsafe_allow_html=True
)

# Auto-refresh
if st.button("Atualizar Dados"):
    st.rerun()

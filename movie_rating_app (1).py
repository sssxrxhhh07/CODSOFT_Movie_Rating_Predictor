#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║         IMDb India — Movie Rating Prediction Web App            ║
║         Retro Brown / Black Theme  •  Flask + Plotly            ║
╚══════════════════════════════════════════════════════════════════╝

Run:
    pip install flask pandas numpy scikit-learn plotly
    python movie_rating_app.py
Then open: http://localhost:5000
"""

# ─── stdlib ──────────────────────────────────────────────────────
import io, base64, warnings, json, os
warnings.filterwarnings("ignore")

# ─── third-party ─────────────────────────────────────────────────
import pandas as pd
import numpy  as np
from flask import Flask, render_template_string, jsonify, request

from sklearn.model_selection   import train_test_split
from sklearn.preprocessing     import LabelEncoder, StandardScaler
from sklearn.ensemble          import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model      import LinearRegression, Ridge
from sklearn.metrics           import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute            import SimpleImputer

import plotly.graph_objects as go
import plotly.express       as px
from   plotly.subplots      import make_subplots

# ══════════════════════════════════════════════════════════════════
# GLOBAL PALETTE  (retro brown / black)
# ══════════════════════════════════════════════════════════════════
PAL = dict(
    bg        = "#0D0A07",
    surface   = "#1A1208",
    card      = "#231A0E",
    border    = "#3D2B14",
    gold      = "#C8960C",
    amber     = "#D4860A",
    cream     = "#E8D5A3",
    muted     = "#7A6040",
    text      = "#F0E0B0",
    red       = "#B84020",
    green     = "#5C8C30",
    teal      = "#3D7A6C",
)
PLOTLY_LAYOUT = dict(
    paper_bgcolor = PAL["card"],
    plot_bgcolor  = PAL["surface"],
    font          = dict(color=PAL["cream"], family="'Courier New', monospace", size=11),
    title_font    = dict(color=PAL["gold"],  family="'Georgia', serif", size=14),
    margin        = dict(l=50, r=20, t=50, b=40),
    colorway      = [PAL["gold"], PAL["amber"], PAL["red"], PAL["green"],
                     PAL["teal"], "#8B6914", "#C05A20", "#9ABC5A"],
    xaxis         = dict(gridcolor=PAL["border"], linecolor=PAL["border"],
                         tickfont=dict(color=PAL["muted"])),
    yaxis         = dict(gridcolor=PAL["border"], linecolor=PAL["border"],
                         tickfont=dict(color=PAL["muted"])),
    hoverlabel    = dict(bgcolor=PAL["card"], bordercolor=PAL["gold"],
                         font_color=PAL["cream"]),
)

CSV_PATH = "IMDb Movies India.csv"

# ══════════════════════════════════════════════════════════════════
# DATA PIPELINE  (runs once at startup)
# ══════════════════════════════════════════════════════════════════
def load_and_train():
    df = pd.read_csv(CSV_PATH, encoding="latin1")

    # ── clean ──
    df["Year"]     = df["Year"].str.extract(r"(\d{4})").astype(float)
    df["Duration"] = df["Duration"].str.extract(r"(\d+)").astype(float)
    df["Votes"]    = pd.to_numeric(df["Votes"], errors="coerce")
    df             = df.dropna(subset=["Rating"])

    # ── feature engineering ──
    df["Primary_Genre"]      = df["Genre"].str.split(",").str[0].str.strip()
    df["Genre_Count"]        = df["Genre"].str.split(",").str.len().fillna(1)
    df["Log_Votes"]          = np.log1p(df["Votes"].fillna(0))
    df["Director_Avg_Rating"]= df.groupby("Director")["Rating"].transform("mean")
    df["Actor1_Avg_Rating"]  = df.groupby("Actor 1")["Rating"].transform("mean")

    le = LabelEncoder()
    df["Genre_Encoded"] = le.fit_transform(df["Primary_Genre"].fillna("Unknown"))

    FEATURES = ["Year","Duration","Log_Votes","Genre_Encoded","Genre_Count",
                 "Director_Avg_Rating","Actor1_Avg_Rating"]

    X = df[FEATURES].copy()
    y = df["Rating"].copy()

    imputer = SimpleImputer(strategy="median")
    X = pd.DataFrame(imputer.fit_transform(X), columns=FEATURES)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler      = StandardScaler()
    X_train_sc  = scaler.fit_transform(X_train)
    X_test_sc   = scaler.transform(X_test)

    models = {
        "Linear Regression" : LinearRegression(),
        "Ridge Regression"  : Ridge(alpha=1.0),
        "Random Forest"     : RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
        "Gradient Boosting" : GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42),
    }

    results = {}
    for name, model in models.items():
        if "Regression" in name:
            model.fit(X_train_sc, y_train); preds = model.predict(X_test_sc)
        else:
            model.fit(X_train, y_train);    preds = model.predict(X_test)
        results[name] = dict(
            rmse  = float(np.sqrt(mean_squared_error(y_test, preds))),
            mae   = float(mean_absolute_error(y_test, preds)),
            r2    = float(r2_score(y_test, preds)),
            preds = preds.tolist(),
        )

    best = max(results, key=lambda k: results[k]["r2"])
    fi   = pd.Series(models["Random Forest"].feature_importances_, index=FEATURES).sort_values(ascending=False)

    return dict(
        df       = df,
        results  = results,
        best     = best,
        fi       = fi,
        y_test   = y_test.tolist(),
        FEATURES = FEATURES,
        le       = le,
        scaler   = scaler,
        imputer  = imputer,
        models   = models,
        genres   = sorted(df["Primary_Genre"].dropna().unique().tolist()),
    )

print("⏳  Loading data & training models …")
STATE = load_and_train()
print(f"✅  Best model: {STATE['best']}  R²={STATE['results'][STATE['best']]['r2']:.4f}")

# ══════════════════════════════════════════════════════════════════
# PLOTLY CHART BUILDERS
# ══════════════════════════════════════════════════════════════════
def fig_to_json(fig):
    fig.update_layout(**PLOTLY_LAYOUT)
    return json.loads(fig.to_json())

def chart_rating_dist():
    df = STATE["df"]
    fig = go.Figure(go.Histogram(
        x=df["Rating"], nbinsx=40,
        marker_color=PAL["amber"], marker_line_color=PAL["gold"],
        marker_line_width=0.5, opacity=0.85,
    ))
    fig.add_vline(x=df["Rating"].mean(), line_color=PAL["cream"],
                  line_dash="dash", line_width=1.5,
                  annotation_text=f"μ={df['Rating'].mean():.2f}",
                  annotation_font_color=PAL["cream"])
    fig.update_layout(title="Rating Distribution", xaxis_title="Rating", yaxis_title="Count")
    return fig_to_json(fig)

def chart_genre_median():
    df   = STATE["df"]
    data = (df.groupby("Primary_Genre")["Rating"].median()
              .sort_values(ascending=True).tail(10))
    fig  = go.Figure(go.Bar(
        y=data.index, x=data.values, orientation="h",
        marker_color=[PAL["gold"], PAL["amber"], PAL["red"], PAL["green"],
                      PAL["teal"], PAL["gold"], PAL["amber"], PAL["red"],
                      PAL["green"], PAL["teal"]],
        text=[f"{v:.2f}" for v in data.values], textposition="outside",
        textfont_color=PAL["cream"],
    ))
    fig.update_layout(title="Median Rating by Genre (Top 10)", xaxis_title="Median Rating")
    return fig_to_json(fig)

def chart_movies_per_year():
    df   = STATE["df"].dropna(subset=["Year"])
    data = df["Year"].value_counts().sort_index()
    fig  = go.Figure(go.Scatter(
        x=data.index, y=data.values, mode="lines",
        fill="tozeroy",
        line=dict(color=PAL["gold"], width=2),
        fillcolor="rgba(200,150,12,0.18)",
    ))
    fig.update_layout(title="Movies Released per Year", xaxis_title="Year", yaxis_title="Count")
    return fig_to_json(fig)

def chart_votes_vs_rating():
    df  = STATE["df"].dropna(subset=["Log_Votes"])
    sample = df.sample(min(3000, len(df)), random_state=1)
    fig = go.Figure(go.Scattergl(
        x=sample["Log_Votes"], y=sample["Rating"],
        mode="markers",
        marker=dict(
            size=4, opacity=0.6,
            color=sample["Rating"],
            colorscale=[[0, PAL["red"]], [0.5, PAL["amber"]], [1, PAL["cream"]]],
            showscale=True,
            colorbar=dict(title=dict(text="Rating", font_color=PAL["cream"]),
                          tickfont_color=PAL["muted"]),
        ),
    ))
    fig.update_layout(title="Log(Votes) vs Rating", xaxis_title="Log(Votes)", yaxis_title="Rating")
    return fig_to_json(fig)

def chart_director_dist():
    df  = STATE["df"]
    fig = go.Figure(go.Histogram(
        x=df["Director_Avg_Rating"], nbinsx=35,
        marker_color=PAL["green"], marker_line_color=PAL["teal"],
        marker_line_width=0.5, opacity=0.85,
    ))
    fig.update_layout(title="Director Avg Rating Distribution",
                      xaxis_title="Avg Rating", yaxis_title="Count")
    return fig_to_json(fig)

def chart_corr_heatmap():
    df = STATE["df"]
    cols = ["Rating","Year","Duration","Log_Votes","Director_Avg_Rating","Actor1_Avg_Rating"]
    corr = df[cols].corr().round(2)
    fig  = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index,
        colorscale=[[0, PAL["red"]], [0.5, PAL["surface"]], [1, PAL["gold"]]],
        zmid=0, text=corr.values,
        texttemplate="%{text}",
        textfont_size=10,
        hoverongaps=False,
    ))
    fig.update_layout(title="Feature Correlation Heatmap")
    return fig_to_json(fig)

def chart_model_rmse():
    res  = STATE["results"]
    names= list(res.keys())
    rmse = [res[n]["rmse"] for n in names]
    fig  = go.Figure(go.Bar(
        y=names, x=rmse, orientation="h",
        marker_color=[PAL["red"], PAL["amber"], PAL["gold"], PAL["green"]],
        text=[f"{v:.4f}" for v in rmse], textposition="outside",
        textfont_color=PAL["cream"],
    ))
    fig.update_layout(title="Model Comparison — RMSE ↓", xaxis_title="RMSE")
    return fig_to_json(fig)

def chart_model_r2():
    res  = STATE["results"]
    names= list(res.keys())
    r2   = [res[n]["r2"] for n in names]
    fig  = go.Figure(go.Bar(
        y=names, x=r2, orientation="h",
        marker_color=[PAL["red"], PAL["amber"], PAL["gold"], PAL["green"]],
        text=[f"{v:.4f}" for v in r2], textposition="outside",
        textfont_color=PAL["cream"],
    ))
    fig.update_layout(title="Model Comparison — R² Score ↑", xaxis_title="R²")
    return fig_to_json(fig)

def chart_feat_importance():
    fi  = STATE["fi"]
    fig = go.Figure(go.Bar(
        y=fi.index, x=fi.values, orientation="h",
        marker_color=[PAL["gold"], PAL["amber"], PAL["teal"], PAL["red"],
                      PAL["green"], PAL["muted"], PAL["muted"]],
        text=[f"{v:.4f}" for v in fi.values], textposition="outside",
        textfont_color=PAL["cream"],
    ))
    fig.update_layout(title="Feature Importance (Random Forest)", xaxis_title="Importance")
    return fig_to_json(fig)

def chart_actual_vs_pred():
    best  = STATE["best"]
    y_test= STATE["y_test"]
    preds = STATE["results"][best]["preds"]
    fig   = go.Figure()
    fig.add_trace(go.Scattergl(
        x=y_test, y=preds, mode="markers",
        marker=dict(size=3, opacity=0.5, color=PAL["amber"]),
        name="Predictions",
    ))
    fig.add_trace(go.Scatter(
        x=[1,10], y=[1,10], mode="lines",
        line=dict(color=PAL["cream"], dash="dash", width=1.5),
        name="Perfect Fit",
    ))
    fig.update_layout(title=f"Actual vs Predicted ({best})",
                      xaxis_title="Actual Rating", yaxis_title="Predicted Rating")
    return fig_to_json(fig)

def chart_residuals():
    best     = STATE["best"]
    y_test   = np.array(STATE["y_test"])
    preds    = np.array(STATE["results"][best]["preds"])
    residuals= y_test - preds
    fig = go.Figure(go.Histogram(
        x=residuals, nbinsx=50,
        marker_color=PAL["red"], marker_line_color=PAL["amber"],
        marker_line_width=0.5, opacity=0.85,
    ))
    fig.add_vline(x=0, line_color=PAL["cream"], line_dash="dash")
    fig.update_layout(title="Residuals Distribution",
                      xaxis_title="Residual", yaxis_title="Count")
    return fig_to_json(fig)

# ══════════════════════════════════════════════════════════════════
# FLASK APP
# ══════════════════════════════════════════════════════════════════
app = Flask(__name__)

# ── HTML template ─────────────────────────────────────────────────
HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>🎬 IMDb India — Movie Rating Predictor</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=Special+Elite&family=Courier+Prime:wght@400;700&display=swap" rel="stylesheet">
<style>
/* ── RESET & GLOBALS ── */
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:      #0D0A07;
  --surface: #1A1208;
  --card:    #231A0E;
  --border:  #3D2B14;
  --gold:    #C8960C;
  --amber:   #D4860A;
  --cream:   #E8D5A3;
  --muted:   #7A6040;
  --text:    #F0E0B0;
  --red:     #B84020;
  --green:   #5C8C30;
  --teal:    #3D7A6C;
}
html{scroll-behavior:smooth}
body{
  background:var(--bg);
  color:var(--text);
  font-family:'Courier Prime', monospace;
  line-height:1.6;
  min-height:100vh;
}

/* ── GRAIN TEXTURE ── */
body::before{
  content:"";position:fixed;inset:0;pointer-events:none;z-index:999;
  background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='300' height='300'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='300' height='300' filter='url(%23n)' opacity='0.04'/%3E%3C/svg%3E");
  opacity:0.35;
}

/* ── HEADER / HERO ── */
.hero{
  position:relative;overflow:hidden;
  background:linear-gradient(180deg,#1A0E00 0%,#0D0A07 100%);
  border-bottom:2px solid var(--border);
  padding:3rem 2rem 2.5rem;
  text-align:center;
}
.hero::after{
  content:"";position:absolute;inset:0;pointer-events:none;
  background:radial-gradient(ellipse 80% 60% at 50% 0%,rgba(200,150,12,0.12) 0%,transparent 70%);
}
.hero-badge{
  display:inline-block;
  border:1px solid var(--border);
  background:var(--card);
  color:var(--muted);
  font-size:.7rem;letter-spacing:.2em;text-transform:uppercase;
  padding:.35rem 1rem;border-radius:2px;margin-bottom:1.2rem;
}
.hero h1{
  font-family:'Playfair Display', serif;
  font-size:clamp(2rem,5vw,4rem);font-weight:900;
  color:var(--gold);text-shadow:0 0 40px rgba(200,150,12,0.4);
  letter-spacing:.02em;line-height:1.1;
}
.hero h1 em{color:var(--cream);font-style:normal;}
.hero p{
  max-width:640px;margin:.8rem auto 0;
  color:var(--muted);font-size:.95rem;line-height:1.7;
}
.hero-stats{
  display:flex;justify-content:center;gap:3rem;
  margin-top:2rem;padding-top:1.5rem;
  border-top:1px solid var(--border);
  flex-wrap:wrap;
}
.hero-stat{text-align:center}
.hero-stat .val{
  font-family:'Playfair Display',serif;
  font-size:1.8rem;font-weight:700;color:var(--gold);
  display:block;
}
.hero-stat .lbl{font-size:.7rem;letter-spacing:.15em;text-transform:uppercase;color:var(--muted);}

/* ── NAV ── */
nav{
  position:sticky;top:0;z-index:100;
  background:rgba(13,10,7,.96);
  backdrop-filter:blur(10px);
  border-bottom:1px solid var(--border);
  display:flex;justify-content:center;gap:0;
  overflow-x:auto;
}
nav a{
  display:inline-block;
  padding:.85rem 1.4rem;
  color:var(--muted);font-size:.8rem;letter-spacing:.12em;
  text-transform:uppercase;text-decoration:none;
  border-bottom:2px solid transparent;
  white-space:nowrap;
  transition:color .2s,border-color .2s;
}
nav a:hover,nav a.active{color:var(--gold);border-bottom-color:var(--gold);}

/* ── LAYOUT ── */
.page{max-width:1280px;margin:0 auto;padding:2.5rem 1.5rem 4rem;}
section{margin-bottom:4rem;}

/* ── SECTION HEADERS ── */
.sec-head{
  display:flex;align-items:center;gap:1rem;
  margin-bottom:1.8rem;
  padding-bottom:.8rem;
  border-bottom:1px solid var(--border);
}
.sec-head h2{
  font-family:'Playfair Display',serif;
  font-size:1.5rem;font-weight:700;color:var(--cream);
  letter-spacing:.02em;
}
.sec-num{
  font-family:'Special Elite',cursive;
  font-size:2.5rem;color:var(--border);line-height:1;
}

/* ── CARDS ── */
.card{
  background:var(--card);
  border:1px solid var(--border);
  border-radius:4px;
  padding:1.2rem;
  position:relative;
}
.card::before{
  content:"";position:absolute;top:0;left:0;right:0;height:2px;
  background:linear-gradient(90deg,var(--border),var(--gold),var(--border));
  border-radius:4px 4px 0 0;
}
.chart-title{
  font-family:'Special Elite',cursive;
  font-size:.8rem;letter-spacing:.12em;text-transform:uppercase;
  color:var(--muted);margin-bottom:.8rem;
}
.plotly-chart{width:100%;height:280px;}

/* ── GRID LAYOUTS ── */
.grid-3{display:grid;grid-template-columns:repeat(3,1fr);gap:1.2rem;}
.grid-2{display:grid;grid-template-columns:repeat(2,1fr);gap:1.2rem;}
.grid-2-1{display:grid;grid-template-columns:2fr 1fr;gap:1.2rem;}
.grid-1-2{display:grid;grid-template-columns:1fr 2fr;gap:1.2rem;}
@media(max-width:900px){
  .grid-3,.grid-2,.grid-2-1,.grid-1-2{grid-template-columns:1fr;}
}

/* ── METRICS TABLE ── */
.metrics-table{width:100%;border-collapse:collapse;font-size:.85rem;}
.metrics-table th{
  color:var(--gold);text-transform:uppercase;letter-spacing:.1em;font-size:.7rem;
  padding:.6rem 1rem;text-align:left;border-bottom:2px solid var(--border);
  background:var(--surface);
}
.metrics-table td{
  padding:.6rem 1rem;border-bottom:1px solid var(--border);
  color:var(--cream);
}
.metrics-table tr:last-child td{border-bottom:none;}
.metrics-table .best td{background:rgba(200,150,12,.08);color:var(--gold);}
.badge{
  display:inline-block;
  padding:.15rem .5rem;border-radius:2px;font-size:.68rem;letter-spacing:.06em;
  text-transform:uppercase;
}
.badge-gold{background:rgba(200,150,12,.2);color:var(--gold);border:1px solid var(--gold);}
.badge-red {background:rgba(184,64,32,.2);color:var(--red); border:1px solid var(--red);}
.badge-green{background:rgba(92,140,48,.2);color:var(--green);border:1px solid var(--green);}

/* ── PREDICTOR FORM ── */
.predictor{
  display:grid;grid-template-columns:1fr 1fr;gap:1.5rem;
}
@media(max-width:700px){.predictor{grid-template-columns:1fr;}}
.form-group{display:flex;flex-direction:column;gap:.4rem;}
.form-group label{
  font-size:.7rem;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);
}
.form-group input,.form-group select{
  background:var(--surface);
  border:1px solid var(--border);
  border-radius:3px;
  color:var(--cream);
  font-family:'Courier Prime',monospace;
  font-size:.9rem;
  padding:.55rem .8rem;
  outline:none;
  transition:border-color .2s;
}
.form-group input:focus,.form-group select:focus{border-color:var(--gold);}
.form-group select option{background:var(--card);}
.btn-predict{
  grid-column:1/-1;
  padding:.85rem;
  background:linear-gradient(135deg,var(--amber),var(--gold));
  border:none;border-radius:3px;
  color:#0D0A07;
  font-family:'Special Elite',cursive;
  font-size:1rem;letter-spacing:.12em;text-transform:uppercase;
  cursor:pointer;
  transition:opacity .2s,transform .1s;
}
.btn-predict:hover{opacity:.9;transform:translateY(-1px);}
.btn-predict:active{transform:translateY(0);}
.pred-result{
  grid-column:1/-1;
  display:none;
  background:var(--surface);
  border:1px solid var(--gold);
  border-radius:4px;
  padding:1.5rem;
  text-align:center;
}
.pred-result.visible{display:block;}
.pred-score{
  font-family:'Playfair Display',serif;
  font-size:3.5rem;font-weight:900;color:var(--gold);
  text-shadow:0 0 30px rgba(200,150,12,.5);
}
.pred-label{
  font-size:.75rem;letter-spacing:.15em;text-transform:uppercase;color:var(--muted);
  margin-top:.3rem;
}
.stars{font-size:1.5rem;margin:.5rem 0;}
.pred-bar{
  margin:.8rem auto 0;max-width:300px;height:6px;
  background:var(--border);border-radius:3px;overflow:hidden;
}
.pred-bar-fill{height:100%;background:linear-gradient(90deg,var(--red),var(--amber),var(--gold));border-radius:3px;transition:width .8s ease;}
.pred-note{font-size:.75rem;color:var(--muted);margin-top:.6rem;}

/* ── INSIGHTS ── */
.insight-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:1rem;}
.insight{
  background:var(--surface);border:1px solid var(--border);
  border-radius:4px;padding:1.2rem;
}
.insight-icon{font-size:1.8rem;margin-bottom:.5rem;}
.insight-title{
  font-size:.7rem;letter-spacing:.12em;text-transform:uppercase;
  color:var(--muted);margin-bottom:.4rem;
}
.insight-val{
  font-family:'Playfair Display',serif;
  font-size:1.4rem;font-weight:700;color:var(--gold);
}
.insight-desc{font-size:.78rem;color:var(--muted);margin-top:.3rem;line-height:1.5;}

/* ── FOOTER ── */
footer{
  border-top:1px solid var(--border);
  padding:1.5rem;text-align:center;
  color:var(--muted);font-size:.75rem;letter-spacing:.08em;
}
.divider{
  display:flex;align-items:center;gap:1rem;
  margin:3rem 0;color:var(--border);font-size:.8rem;
}
.divider::before,.divider::after{content:"";flex:1;height:1px;background:var(--border);}

/* ── LOADING ── */
.spinner{
  display:inline-block;width:18px;height:18px;
  border:2px solid var(--border);border-top-color:var(--gold);
  border-radius:50%;animation:spin .7s linear infinite;vertical-align:middle;margin-right:.5rem;
}
@keyframes spin{to{transform:rotate(360deg)}}

/* ── SCROLLBAR ── */
::-webkit-scrollbar{width:6px;height:6px}
::-webkit-scrollbar-track{background:var(--bg)}
::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px}
::-webkit-scrollbar-thumb:hover{background:var(--muted)}
</style>
</head>
<body>

<!-- ── HERO ─────────────────────────────────────────────────── -->
<header class="hero">
  <div class="hero-badge">Data Science Project  •  IMDb India</div>
  <h1>🎬 Movie Rating <em>Prediction</em></h1>
  <p>Explore historical Bollywood data, uncover what makes a great film, and predict ratings using regression machine learning models.</p>
  <div class="hero-stats" id="heroStats">
    <div class="hero-stat"><span class="val" id="hs-movies">—</span><span class="lbl">Movies</span></div>
    <div class="hero-stat"><span class="val" id="hs-mean">—</span><span class="lbl">Mean Rating</span></div>
    <div class="hero-stat"><span class="val" id="hs-best">—</span><span class="lbl">Best Model R²</span></div>
    <div class="hero-stat"><span class="val" id="hs-rmse">—</span><span class="lbl">Best RMSE</span></div>
  </div>
</header>

<!-- ── NAV ──────────────────────────────────────────────────── -->
<nav>
  <a href="#overview"  class="active">Overview</a>
  <a href="#eda">Exploration</a>
  <a href="#features">Features</a>
  <a href="#models">Models</a>
  <a href="#predict">Predict</a>
</nav>

<!-- ── MAIN ─────────────────────────────────────────────────── -->
<main class="page">

  <!-- ── OVERVIEW ── -->
  <section id="overview">
    <div class="sec-head"><span class="sec-num">01</span><h2>Dataset Overview</h2></div>
    <div class="insight-grid" id="insightGrid">
      <div class="insight"><div class="insight-icon">🎞</div><div class="insight-title">Total Movies</div><div class="insight-val" id="ins-total">…</div><div class="insight-desc">Indian films with at least one rating</div></div>
      <div class="insight"><div class="insight-icon">⭐</div><div class="insight-title">Average Rating</div><div class="insight-val" id="ins-mean">…</div><div class="insight-desc">Mean IMDb user rating across the dataset</div></div>
      <div class="insight"><div class="insight-icon">🎭</div><div class="insight-title">Unique Genres</div><div class="insight-val" id="ins-genres">…</div><div class="insight-desc">Primary genres represented</div></div>
      <div class="insight"><div class="insight-icon">🎬</div><div class="insight-title">Directors</div><div class="insight-val" id="ins-dirs">…</div><div class="insight-desc">Unique directors in the dataset</div></div>
      <div class="insight"><div class="insight-icon">📅</div><div class="insight-title">Year Span</div><div class="insight-val" id="ins-years">…</div><div class="insight-desc">Earliest to most recent release year</div></div>
      <div class="insight"><div class="insight-icon">🗳</div><div class="insight-title">Median Votes</div><div class="insight-val" id="ins-votes">…</div><div class="insight-desc">Median number of IMDb votes per movie</div></div>
    </div>
    <div class="divider">Exploration</div>
    <div class="grid-3" style="margin-bottom:1.2rem">
      <div class="card"><div class="chart-title">Rating Distribution</div><div id="c-dist" class="plotly-chart"></div></div>
      <div class="card"><div class="chart-title">Movies per Year</div><div id="c-year" class="plotly-chart"></div></div>
      <div class="card"><div class="chart-title">Median Rating by Genre</div><div id="c-genre" class="plotly-chart"></div></div>
    </div>
  </section>

  <!-- ── EDA ── -->
  <section id="eda">
    <div class="sec-head"><span class="sec-num">02</span><h2>Exploratory Analysis</h2></div>
    <div class="grid-2" style="margin-bottom:1.2rem">
      <div class="card"><div class="chart-title">Log(Votes) vs Rating</div><div id="c-votes" class="plotly-chart" style="height:320px"></div></div>
      <div class="card"><div class="chart-title">Director Avg Rating Distribution</div><div id="c-dir" class="plotly-chart" style="height:320px"></div></div>
    </div>
    <div class="card">
      <div class="chart-title">Feature Correlation Heatmap</div>
      <div id="c-corr" class="plotly-chart" style="height:380px"></div>
    </div>
  </section>

  <!-- ── FEATURES ── -->
  <section id="features">
    <div class="sec-head"><span class="sec-num">03</span><h2>Feature Engineering</h2></div>
    <div class="grid-1-2">
      <div class="card" style="align-self:start">
        <div class="chart-title">Engineered Features</div>
        <table class="metrics-table">
          <thead><tr><th>Feature</th><th>Source</th></tr></thead>
          <tbody>
            <tr><td>Year</td><td style="color:var(--muted)">Parsed from "(YYYY)"</td></tr>
            <tr><td>Duration</td><td style="color:var(--muted)">Stripped "min" suffix</td></tr>
            <tr><td>Log_Votes</td><td style="color:var(--muted)">log1p(Votes) — skew fix</td></tr>
            <tr><td>Genre_Encoded</td><td style="color:var(--muted)">LabelEncoder on primary genre</td></tr>
            <tr><td>Genre_Count</td><td style="color:var(--muted)">Number of genres listed</td></tr>
            <tr><td>Director_Avg_Rating</td><td style="color:var(--muted)">Historical director mean</td></tr>
            <tr><td>Actor1_Avg_Rating</td><td style="color:var(--muted)">Historical lead actor mean</td></tr>
          </tbody>
        </table>
      </div>
      <div class="card">
        <div class="chart-title">Feature Importance (Random Forest)</div>
        <div id="c-fi" class="plotly-chart" style="height:300px"></div>
      </div>
    </div>
  </section>

  <!-- ── MODELS ── -->
  <section id="models">
    <div class="sec-head"><span class="sec-num">04</span><h2>Model Performance</h2></div>
    <div class="grid-2" style="margin-bottom:1.2rem">
      <div class="card"><div class="chart-title">RMSE Comparison (lower = better)</div><div id="c-rmse" class="plotly-chart"></div></div>
      <div class="card"><div class="chart-title">R² Score Comparison (higher = better)</div><div id="c-r2" class="plotly-chart"></div></div>
    </div>
    <div class="grid-2" style="margin-bottom:1.2rem">
      <div class="card"><div class="chart-title">Actual vs Predicted</div><div id="c-avp" class="plotly-chart"></div></div>
      <div class="card"><div class="chart-title">Residuals Distribution</div><div id="c-res" class="plotly-chart"></div></div>
    </div>
    <div class="card">
      <div class="chart-title">Full Metrics Summary</div>
      <table class="metrics-table" id="metricsTable"></table>
    </div>
  </section>

  <!-- ── PREDICTOR ── -->
  <section id="predict">
    <div class="sec-head"><span class="sec-num">05</span><h2>Predict a Movie Rating</h2></div>
    <div class="card">
      <p style="color:var(--muted);font-size:.85rem;margin-bottom:1.5rem;">
        Fill in the details below and the best-performing model will estimate the IMDb rating.
      </p>
      <div class="predictor">
        <div class="form-group">
          <label>Release Year</label>
          <input type="number" id="f-year" value="2022" min="1950" max="2025">
        </div>
        <div class="form-group">
          <label>Duration (minutes)</label>
          <input type="number" id="f-duration" value="130" min="30" max="300">
        </div>
        <div class="form-group">
          <label>Expected Votes (approx.)</label>
          <input type="number" id="f-votes" value="500" min="1">
        </div>
        <div class="form-group">
          <label>Primary Genre</label>
          <select id="f-genre"></select>
        </div>
        <div class="form-group">
          <label>Number of Genres Listed</label>
          <input type="number" id="f-gcount" value="2" min="1" max="6">
        </div>
        <div class="form-group">
          <label>Director's Historical Avg Rating</label>
          <input type="number" id="f-diravg" value="6.0" step="0.1" min="1" max="10">
        </div>
        <div class="form-group">
          <label>Lead Actor's Historical Avg Rating</label>
          <input type="number" id="f-actoravg" value="5.8" step="0.1" min="1" max="10">
        </div>

        <button class="btn-predict" onclick="predict()">⚙ Predict Rating</button>

        <div class="pred-result" id="predResult">
          <div class="pred-score" id="predScore">—</div>
          <div class="stars" id="predStars"></div>
          <div class="pred-label">Predicted IMDb Rating</div>
          <div class="pred-bar"><div class="pred-bar-fill" id="predBarFill" style="width:0%"></div></div>
          <div class="pred-note" id="predNote"></div>
        </div>
      </div>
    </div>
  </section>

</main>

<footer>
  <p>🎬  IMDb India · Movie Rating Prediction  ·  Data Science Project<br>
  <span style="opacity:.5">Built with Python · Flask · Scikit-learn · Plotly</span></p>
</footer>

<script>
/* ── fetch chart data & render ── */
const cfg = {responsive:true, displayModeBar:false};

async function loadCharts(){
  const r = await fetch('/api/charts'); 
  const d = await r.json();

  const apply = (id, data) =>
    Plotly.newPlot(id, data.data, data.layout, cfg);

  apply('c-dist',  d.rating_dist);
  apply('c-year',  d.movies_year);
  apply('c-genre', d.genre_median);
  apply('c-votes', d.votes_rating);
  apply('c-dir',   d.director_dist);
  apply('c-corr',  d.corr_heat);
  apply('c-fi',    d.feat_imp);
  apply('c-rmse',  d.model_rmse);
  apply('c-r2',    d.model_r2);
  apply('c-avp',   d.actual_pred);
  apply('c-res',   d.residuals);
}

async function loadStats(){
  const r = await fetch('/api/stats');
  const d = await r.json();

  // hero
  document.getElementById('hs-movies').textContent = d.total.toLocaleString();
  document.getElementById('hs-mean'  ).textContent = d.mean_rating.toFixed(2);
  document.getElementById('hs-best'  ).textContent = d.best_r2.toFixed(4);
  document.getElementById('hs-rmse'  ).textContent = d.best_rmse.toFixed(4);

  // insights
  document.getElementById('ins-total' ).textContent = d.total.toLocaleString();
  document.getElementById('ins-mean'  ).textContent = d.mean_rating.toFixed(2);
  document.getElementById('ins-genres').textContent = d.genre_count;
  document.getElementById('ins-dirs'  ).textContent = d.director_count.toLocaleString();
  document.getElementById('ins-years' ).textContent = d.year_range;
  document.getElementById('ins-votes' ).textContent = d.median_votes.toLocaleString();

  // metrics table
  const tbody = document.getElementById('metricsTable');
  let html = '<thead><tr><th>Model</th><th>RMSE</th><th>MAE</th><th>R²</th><th>Rank</th></tr></thead><tbody>';
  d.model_metrics.forEach((m, i) => {
    const isBest = m.name === d.best_model;
    html += `<tr class="${isBest?'best':''}">
      <td>${m.name}${isBest?' <span class="badge badge-gold">Best</span>':''}</td>
      <td>${m.rmse.toFixed(4)}</td>
      <td>${m.mae.toFixed(4)}</td>
      <td>${m.r2.toFixed(4)}</td>
      <td><span class="badge ${i===0?'badge-gold':i===1?'badge-green':'badge-red'}">#${i+1}</span></td>
    </tr>`;
  });
  html += '</tbody>';
  tbody.innerHTML = html;

  // genre dropdown
  const sel = document.getElementById('f-genre');
  d.genres.forEach(g => {
    const o = document.createElement('option');
    o.value = g; o.textContent = g; sel.appendChild(o);
  });
}

async function predict(){
  const body = {
    year:      parseFloat(document.getElementById('f-year').value),
    duration:  parseFloat(document.getElementById('f-duration').value),
    votes:     parseFloat(document.getElementById('f-votes').value),
    genre:     document.getElementById('f-genre').value,
    gcount:    parseFloat(document.getElementById('f-gcount').value),
    diravg:    parseFloat(document.getElementById('f-diravg').value),
    actoravg:  parseFloat(document.getElementById('f-actoravg').value),
  };

  const btn = document.querySelector('.btn-predict');
  btn.innerHTML = '<span class="spinner"></span>Predicting…';
  btn.disabled = true;

  try{
    const r   = await fetch('/api/predict', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
    const d   = await r.json();
    const rating = d.rating.toFixed(2);
    const pct    = ((d.rating - 1) / 9) * 100;
    const stars  = Math.round(d.rating / 2);
    
    document.getElementById('predScore').textContent = rating + ' / 10';
    document.getElementById('predStars').textContent = '★'.repeat(stars) + '☆'.repeat(5 - stars);
    document.getElementById('predBarFill').style.width = Math.min(pct, 100) + '%';
    document.getElementById('predNote').textContent =
      `Estimated by ${d.model} (R² = ${d.r2.toFixed(4)})`;

    const res = document.getElementById('predResult');
    res.classList.add('visible');
    res.scrollIntoView({behavior:'smooth', block:'nearest'});
  } finally {
    btn.innerHTML = '⚙ Predict Rating';
    btn.disabled = false;
  }
}

/* ── sticky nav highlight ── */
const sections = document.querySelectorAll('section[id]');
const navLinks  = document.querySelectorAll('nav a');
window.addEventListener('scroll', () => {
  let cur = '';
  sections.forEach(s => { if(window.scrollY >= s.offsetTop - 120) cur = s.id; });
  navLinks.forEach(a => { a.classList.toggle('active', a.getAttribute('href') === '#'+cur); });
});

loadStats();
loadCharts();
</script>
</body>
</html>
"""

# ── ROUTES ────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/api/stats")
def api_stats():
    df  = STATE["df"]
    res = STATE["results"]
    best= STATE["best"]

    sorted_models = sorted(res.items(), key=lambda x: x[1]["r2"], reverse=True)
    metrics = [{"name": n, "rmse": v["rmse"], "mae": v["mae"], "r2": v["r2"]}
               for n, v in sorted_models]

    yr = df["Year"].dropna()
    return jsonify(dict(
        total          = len(df),
        mean_rating    = float(df["Rating"].mean()),
        genre_count    = int(df["Primary_Genre"].nunique()),
        director_count = int(df["Director"].nunique()),
        year_range     = f"{int(yr.min())}–{int(yr.max())}",
        median_votes   = int(df["Votes"].median()),
        best_r2        = res[best]["r2"],
        best_rmse      = res[best]["rmse"],
        best_model     = best,
        model_metrics  = metrics,
        genres         = STATE["genres"],
    ))

@app.route("/api/charts")
def api_charts():
    return jsonify(dict(
        rating_dist   = chart_rating_dist(),
        movies_year   = chart_movies_per_year(),
        genre_median  = chart_genre_median(),
        votes_rating  = chart_votes_vs_rating(),
        director_dist = chart_director_dist(),
        corr_heat     = chart_corr_heatmap(),
        feat_imp      = chart_feat_importance(),
        model_rmse    = chart_model_rmse(),
        model_r2      = chart_model_r2(),
        actual_pred   = chart_actual_vs_pred(),
        residuals     = chart_residuals(),
    ))

@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.json
    best = STATE["best"]
    model   = STATE["models"][best]
    scaler  = STATE["scaler"]
    imputer = STATE["imputer"]
    le      = STATE["le"]
    FEATURES= STATE["FEATURES"]

    genre = data.get("genre", "Drama")
    try:   genre_enc = int(le.transform([genre])[0])
    except: genre_enc = 0

    row = pd.DataFrame([{
        "Year"               : data.get("year", 2020),
        "Duration"           : data.get("duration", 120),
        "Log_Votes"          : np.log1p(data.get("votes", 100)),
        "Genre_Encoded"      : genre_enc,
        "Genre_Count"        : data.get("gcount", 2),
        "Director_Avg_Rating": data.get("diravg", 6.0),
        "Actor1_Avg_Rating"  : data.get("actoravg", 5.8),
    }], columns=FEATURES)

    row = pd.DataFrame(imputer.transform(row), columns=FEATURES)

    if "Regression" in best:
        row_sc = scaler.transform(row)
        pred   = float(model.predict(row_sc)[0])
    else:
        pred = float(model.predict(row)[0])

    pred = max(1.0, min(10.0, pred))
    return jsonify(dict(rating=round(pred, 2), model=best,
                        r2=STATE["results"][best]["r2"]))


# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "═"*60)
    print("  🎬  Movie Rating Prediction App")
    print("  🌐  Open: http://localhost:5000")
    print("═"*60 + "\n")
    app.run(debug=True, port=5000)

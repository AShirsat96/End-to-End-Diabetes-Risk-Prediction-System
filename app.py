import json
import os
import boto3

from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

# ============================================================
# AWS CONFIGURATION (FROM ENVIRONMENT VARIABLES)
# ============================================================

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
SAGEMAKER_ENDPOINT = os.getenv("SAGEMAKER_ENDPOINT")

if not SAGEMAKER_ENDPOINT:
    raise RuntimeError("SAGEMAKER_ENDPOINT environment variable is not set")

runtime = boto3.client(
    "sagemaker-runtime",
    region_name=AWS_REGION
)

s3 = boto3.client("s3")

# ============================================================
# LOAD PRECOMPUTED STATS FROM S3 (FOR RECOMMENDATIONS)
# ============================================================

STATS_BUCKET = "diabetes-group-project-bucket"
STATS_KEY = "stats.json"

obj = s3.get_object(Bucket=STATS_BUCKET, Key=STATS_KEY)
STATS = json.loads(obj["Body"].read())

# ============================================================
# DASH APP INITIALIZATION
# ============================================================

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server  # Required for Render + gunicorn

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def generate_recommendations(inputs, risk_score, stats):
    recs = []

    bmi = inputs["BMI"]
    if bmi >= 30:
        recs.append({
            "title": "Weight Management",
            "detail": f"Obesity is associated with ~{stats['bmi']['obese']:.1f}% diabetes prevalence.",
            "action": "A 5–10% reduction in body weight can significantly reduce risk.",
            "priority": "high"
        })
    elif bmi >= 25:
        recs.append({
            "title": "Weight Management",
            "detail": "BMI falls in the overweight range.",
            "action": "Maintain balanced diet and regular physical activity.",
            "priority": "medium"
        })

    if inputs["HighBP"] == 1:
        recs.append({
            "title": "Blood Pressure",
            "detail": f"Diabetes prevalence with high BP: {stats['highbp']['yes']:.1f}%.",
            "action": "Monitor blood pressure and follow treatment guidelines.",
            "priority": "high"
        })

    if inputs["HighChol"] == 1:
        recs.append({
            "title": "Cholesterol",
            "detail": f"High cholesterol correlates with increased diabetes risk.",
            "action": "Reduce saturated fats and increase fiber intake.",
            "priority": "high"
        })

    if inputs["GenHlth"] >= 4:
        recs.append({
            "title": "General Health",
            "detail": f"Poor self-reported health shows elevated diabetes prevalence.",
            "action": "Schedule a comprehensive health evaluation.",
            "priority": "medium"
        })

    if risk_score >= 50:
        recs.append({
            "title": "Screening Recommendation",
            "detail": "Predicted diabetes risk is elevated.",
            "action": "Consider HbA1c testing and clinical follow-up.",
            "priority": "high"
        })

    if not recs:
        recs.append({
            "title": "Risk Assessment",
            "detail": "Risk factors appear well-controlled.",
            "action": "Continue routine health maintenance.",
            "priority": "low"
        })

    priority_order = {"high": 0, "medium": 1, "low": 2}
    recs.sort(key=lambda x: priority_order[x["priority"]])

    return recs


def render_recommendations(recs):
    color_map = {"high": "danger", "medium": "warning", "low": "success"}

    cards = []
    for r in recs:
        cards.append(
            dbc.Card(
                dbc.CardBody([
                    html.H6(r["title"]),
                    html.P(r["detail"], className="text-muted mb-1"),
                    html.P([html.Strong("Action: "), r["action"]], className="mb-0")
                ]),
                outline=True,
                color=color_map[r["priority"]],
                className="mb-2"
            )
        )

    return html.Div([
        html.H5("Clinical Recommendations", className="mt-3"),
        html.Hr(),
        *cards
    ])

# ============================================================
# APP LAYOUT (RESPONSIVE)
# ============================================================

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(
            html.H1(
                "🩺 Diabetes Risk Assessment Dashboard",
                className="text-center py-3",
                style={
                    "backgroundColor": "#1a365d",
                    "color": "white",
                    "borderRadius": "10px",
                    "fontSize": "clamp(1.4rem, 4vw, 2.4rem)"
                }
            )
        )
    ], className="mb-3"),

    dbc.Row([
        # ---------------- INPUT PANEL ----------------
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("Patient Information")),
                dbc.CardBody([
                    dbc.Label("General Health (1–5)"),
                    dcc.Dropdown(
                        id="genhlth",
                        options=[
                            {"label": "Excellent", "value": 1},
                            {"label": "Very Good", "value": 2},
                            {"label": "Good", "value": 3},
                            {"label": "Fair", "value": 4},
                            {"label": "Poor", "value": 5},
                        ],
                        value=2,
                        className="mb-2"
                    ),

                    dbc.Label("BMI"),
                    dbc.Input(id="bmi", type="number", value=25, className="mb-2"),

                    dbc.Label("Age Group"),
                    dcc.Dropdown(
                        id="age",
                        options=[
                            {"label": "18–24", "value": 1},
                            {"label": "25–29", "value": 2},
                            {"label": "30–34", "value": 3},
                            {"label": "35–39", "value": 4},
                            {"label": "40–44", "value": 5},
                            {"label": "45–49", "value": 6},
                            {"label": "50–54", "value": 7},
                            {"label": "55–59", "value": 8},
                            {"label": "60–64", "value": 9},
                            {"label": "65–69", "value": 10},
                            {"label": "70–74", "value": 11},
                            {"label": "75–79", "value": 12},
                            {"label": "80+", "value": 13},
                        ],
                        value=6,
                        className="mb-2"
                    ),

                    dbc.Label("High Blood Pressure"),
                    dcc.Dropdown(
                        id="highbp",
                        options=[{"label": "No", "value": 0}, {"label": "Yes", "value": 1}],
                        value=0,
                        className="mb-2"
                    ),

                    dbc.Label("High Cholesterol"),
                    dcc.Dropdown(
                        id="highchol",
                        options=[{"label": "No", "value": 0}, {"label": "Yes", "value": 1}],
                        value=0,
                        className="mb-3"
                    ),

                    dbc.Button(
                        "Calculate Risk",
                        id="submit",
                        color="success",
                        className="w-100"
                    ),

                    html.P(
                        "Tip: Scroll down to view results",
                        className="text-muted d-md-none text-center mt-2"
                    )
                ])
            ])
        ], xs=12, md=4, className="mb-3"),

        # ---------------- RESULTS PANEL ----------------
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("Risk Assessment Results")),
                dbc.CardBody([
                    html.Div(id="risk-output", className="mb-3"),
                    dcc.Graph(id="risk-gauge"),
                    html.Div(id="recommendations")
                ])
            ])
        ], xs=12, md=8)
    ])
], fluid=True, className="p-3")

# ============================================================
# CALLBACK
# ============================================================

@app.callback(
    [Output("risk-output", "children"),
     Output("risk-gauge", "figure"),
     Output("recommendations", "children")],
    Input("submit", "n_clicks"),
    State("genhlth", "value"),
    State("bmi", "value"),
    State("age", "value"),
    State("highbp", "value"),
    State("highchol", "value")
)
def predict_risk(n_clicks, genhlth, bmi, age, highbp, highchol):
    if not n_clicks:
        return "", {}, ""

    payload = {
        "GenHlth": genhlth,
        "BMI": bmi,
        "Age": age,
        "HighBP": highbp,
        "HighChol": highchol,
        "CholCheck": 1,
        "DiffWalk": 0,
        "HeartDiseaseorAttack": 0,
        "Sex": 1,
        "HvyAlcoholConsump": 0
    }

    response = runtime.invoke_endpoint(
        EndpointName=SAGEMAKER_ENDPOINT,
        ContentType="application/json",
        Body=json.dumps(payload)
    )

    result = json.loads(response["Body"].read())
    score = result["risk_score"]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix": "%"},
        gauge={
            "axis": {
                "range": [0, 100],
                "tickwidth": 1,
                "tickcolor": "darkblue",
            },
            "bar": {"color": "#22c55e"},
            "steps": [
                {"range": [0, 25], "color": "#dcfce7"},
                {"range": [25, 50], "color": "#fef3c7"},
                {"range": [50, 75], "color": "#fed7aa"},
                {"range": [75, 100], "color": "#fecaca"},
            ],
        }
    ))

    fig.update_layout(height=260, margin=dict(t=30, b=20, l=20, r=20))

    recs = generate_recommendations(payload, score, STATS)

    return (
        f"Predicted Diabetes Risk: {score:.1f}%",
        fig,
        render_recommendations(recs)
    )

# ============================================================
# LOCAL RUN (NOT USED ON RENDER)
# ============================================================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=True)

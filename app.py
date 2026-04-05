from flask import Flask, render_template, request, jsonify, send_file, session, redirect, url_for
import numpy as np
import requests
from tensorflow.keras.models import load_model

# 🔥 NEW IMPORTS
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import io

# 🔥 NEW IMPORT
import matplotlib.pyplot as plt

app = Flask(__name__)

# 🔥 REQUIRED FOR SESSION
app.secret_key = "secret123"

model = load_model("model.h5", compile=False, safe_mode=False)



# ✅ HOME (UPDATED SLIGHTLY - SAFE)
@app.route('/')
def home():
    data = session.pop('result', None)

    return render_template(
        'index.html',
        prediction_text=data["prediction_text"] if data else None,
        prediction_value=data["prediction_value"] if data else None,
        forecast_data=data["forecast_data"] if data else None,
        forecast_labels=data["forecast_labels"] if data else None
    )


# 🔮 PREDICTION (ML MODEL)
@app.route("/predict", methods=["POST"])
def predict():

    print("🔥 PREDICT ROUTE HIT")

    features = [
        float(request.form["temp"]),
        float(request.form["humidity"]),
        float(request.form["pressure"]),
        float(request.form["zenith"]),
        float(request.form["cloud"]),
        float(request.form["wind"]),
        float(request.form["radiation"])
    ]

    print("📊 FEATURES:", features)

    final = np.array(features).reshape(1, -1)

    prediction = float(abs(model.predict(final)[0][0]))
    prediction = round(prediction, 2)

    print("⚡ PREDICTION:", prediction)

    user_id = request.form.get("user_id", "anonymous")
    print("👤 USER ID:", user_id)

    # forecast
    hours = list(range(6, 18))

    forecast = []
    for h in hours:
        peak = np.exp(-((h - 12) ** 2) / 10)
        value = prediction * peak
        forecast.append(float(round(value, 2)))

    # 🔥 STORE FOR PDF (unchanged)
    session['last_data'] = {
        "features": features,
        "prediction": prediction,
        "forecast": forecast
    }

    # ✅ NEW: STORE FOR REDIRECT (THIS FIXES BACK BUTTON ISSUE)
    session['result'] = {
        "prediction_text": f"Predicted Solar Power: {prediction} kW",
        "prediction_value": prediction,
        "forecast_data": forecast,
        "forecast_labels": [f"{h}:00" for h in hours]
    }

    # ✅ REDIRECT INSTEAD OF RENDER
    return redirect(url_for('home'))


# GAUGE PAGE (UNCHANGED)
@app.route("/gauge/<value>")
def gauge(value):
    return render_template("gauge.html", value=float(value))


# 🌤 FORECAST API (UNCHANGED)
@app.route("/api/forecast")
def api_forecast():

    lat, lon = 28.61, 77.20

    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=shortwave_radiation&timezone=auto"

    try:
        response = requests.get(url)
        data = response.json()

        radiation = data["hourly"]["shortwave_radiation"]

        solar = radiation[:12]

        forecast = [round(v / 1000, 3) for v in solar]

        labels = [f"{i+6}:00" for i in range(len(forecast))]

        return jsonify({
            "labels": labels,
            "data": forecast
        })

    except Exception as e:
        return jsonify({
            "labels": [],
            "data": [],
            "error": str(e)
        })


# FORECAST PAGE (UNCHANGED)
@app.route("/forecast")
def forecast():
    return render_template("forecast.html")


# 🔥 PDF DOWNLOAD (UNCHANGED)
@app.route("/download_pdf")
def download_pdf():

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()

    content = []

    data = session.get('last_data')

    # 🔵 HEADER
    content.append(Paragraph("<b>☀ SolarPulse AI Report</b>", styles["Title"]))
    content.append(Spacer(1, 6))
    content.append(Paragraph("AI-powered Solar Energy Prediction Dashboard", styles["Normal"]))
    content.append(Spacer(1, 20))

    if data:

        features = data["features"]
        prediction = data["prediction"]
        forecast = data["forecast"]

        labels = [
            "Temperature", "Humidity", "Pressure",
            "Zenith", "Cloud", "Wind", "Radiation"
        ]

        # =========================
        # 🟡 INPUT PARAMETERS (CLEAN)
        # =========================
        content.append(Paragraph("<b>Input Parameters</b>", styles["Heading2"]))
        content.append(Spacer(1, 10))

        for i in range(len(features)):
            content.append(
                Paragraph(f"{labels[i]}: {round(features[i],2)}", styles["Normal"])
            )

        content.append(Spacer(1, 20))

        # =========================
        # 📊 INPUT GRAPH
        # =========================
        plt.figure(figsize=(6,3))
        plt.bar(labels, features, color="#5DADE2")
        plt.xticks(rotation=25)
        plt.title("Input Feature Distribution")
        plt.tight_layout()

        input_buffer = io.BytesIO()
        plt.savefig(input_buffer, format='png')
        plt.close()
        input_buffer.seek(0)

        content.append(Paragraph("<b>Input Feature Graph</b>", styles["Heading3"]))
        content.append(Spacer(1, 10))
        content.append(Image(input_buffer, width=420, height=220))
        content.append(Spacer(1, 20))

        # =========================
        # 🔥 PREDICTION
        # =========================
        content.append(Paragraph("<b>Predicted Solar Power</b>", styles["Heading2"]))
        content.append(Spacer(1, 10))
        content.append(Paragraph(f"<b>{prediction} kW</b>", styles["Title"]))
        content.append(Spacer(1, 20))

        # =========================
        # 🟢 OUTPUT BAR GRAPH (FIXED)
        # =========================
        plt.figure(figsize=(4,3))

        bars = plt.bar(
            ["Output"],
            [prediction],
            color="#27AE60",
            width=0.5
        )

        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width()/2,
                height + 1,
                f"{round(height,2)} kW",
                ha='center',
                fontweight='bold'
            )

        plt.title("Solar Output (kW)")
        plt.ylabel("Power")
        plt.grid(axis='y', linestyle='--', alpha=0.4)
        plt.tight_layout()

        output_buffer = io.BytesIO()
        plt.savefig(output_buffer, format='png')
        plt.close()
        output_buffer.seek(0)

        content.append(Paragraph("<b>Predicted Output Graph</b>", styles["Heading3"]))
        content.append(Spacer(1, 10))
        content.append(Image(output_buffer, width=300, height=200))
        content.append(Spacer(1, 20))

        # =========================
        # 📈 24 HOUR FORECAST
        # =========================
        plt.figure(figsize=(6,3))
        plt.plot(forecast, marker='o', color="#E67E22")
        plt.title("24 Hour Solar Forecast")
        plt.xlabel("Hours")
        plt.ylabel("Power (kW)")
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout()

        forecast_buffer = io.BytesIO()
        plt.savefig(forecast_buffer, format='png')
        plt.close()
        forecast_buffer.seek(0)

        content.append(Paragraph("<b>24 Hour Forecast</b>", styles["Heading3"]))
        content.append(Spacer(1, 10))
        content.append(Image(forecast_buffer, width=420, height=220))
        content.append(Spacer(1, 20))

        # =========================
        # 🟣 FOOTER
        # =========================
        content.append(Paragraph(
            "Generated using AI-based Solar Prediction Model | SolarPulse AI",
            styles["Normal"]
        ))

    else:
        content.append(Paragraph("No data available", styles["Normal"]))

    doc.build(content)

    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name="solar_report.pdf",
        mimetype='application/pdf'
    )


# RUN
if __name__ == "__main__":
    app.run(debug=True)

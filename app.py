from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import pandas as pd

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Load the trained model and preprocessors
with open("pistol_recommender.pkl", "rb") as model_file:
    model = pickle.load(model_file)
with open("label_encoder.pkl", "rb") as le_file:
    label_encoder = pickle.load(le_file)
with open("feature_selector.pkl", "rb") as fs_file:
    selector = pickle.load(fs_file)
with open("scaler(2).pkl", "rb") as scaler_file:  # Ensure correct file name
    scaler = pickle.load(scaler_file)

# Load the dataset for reference
file_path = "Updated_Indian_Army_Pistols_Corrected.csv"
df = pd.read_csv(file_path)

# Essential columns
pistol_name_column = "Pistol Name"
label = "Best Use Case"

# **Ensure dataset labels are encoded**
df[label] = label_encoder.transform(df[label])

# **Function to Recommend Pistol**
def recommend_pistol(mission_type):
    if mission_type not in label_encoder.classes_:
        return {"error": "Invalid Mission Type. Please enter a valid mission."}

    # Encode mission type to match dataset encoding
    mission_code = label_encoder.transform([mission_type])[0]

    # Filter pistols based on encoded mission type
    filtered_pistols = df[df[label] == mission_code]

    if filtered_pistols.empty:
        return {"error": f"No suitable pistol found for {mission_type}."}

    best_pistol = (
        filtered_pistols.iloc[0][pistol_name_column]
        if pistol_name_column in filtered_pistols.columns
        else "No Name Available"
    )
    return {"Recommended Pistol": best_pistol}

# **Routes**
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/recommend", methods=["POST"])
def get_recommendation():
    try:
        data = request.get_json()
        mission_type = data.get("mission_type")

        if not mission_type:
            return jsonify({"error": "Mission type is required."}), 400

        response = recommend_pistol(mission_type)
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

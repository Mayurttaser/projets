from flask import Flask, render_template, request
import pickle
import pandas as pd

# Démarrer l'application Flask
app = Flask(__name__)

# Charger le modèle RandomForest sauvegardé
model = pickle.load(open("loan_default_rf_model.pkl", "rb"))

# Fonction de prédiction du modèle
def model_pred(features):
    test_data = pd.DataFrame([features])
    prediction = model.predict(test_data)
    return int(prediction[0])

# Route pour la page d'accueil
@app.route("/", methods=["GET"])
def Home():
    return render_template("index.html")

# Route pour la prédiction
@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        # Récupérer les données de l'utilisateur
        credit_lines_outstanding = int(request.form["credit_lines_outstanding"])
        loan_amt_outstanding = float(request.form["loan_amt_outstanding"])
        total_debt_outstanding = float(request.form["total_debt_outstanding"])
        income = float(request.form["income"])
        years_employed = int(request.form["years_employed"])
        fico_score = int(request.form["fico_score"])

        # Faire une prédiction avec le modèle
        prediction = model.predict(
            [[credit_lines_outstanding, loan_amt_outstanding, total_debt_outstanding, income, years_employed, fico_score]]
        )

        # Afficher le résultat de la prédiction
        if prediction[0] == 1:
            return render_template(
                "index.html",
                prediction_text="Risque élevé : Le client pourrait être en défaut de crédit !",
            )

        else:
            return render_template(
                "index.html", prediction_text="Risque faible : Le client n'est pas en défaut de crédit !"
            )

    else:
        return render_template("index.html")

# Exécuter l'application
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)


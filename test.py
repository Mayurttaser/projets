from app import model_pred

# Le nouvel ensemble de données à utiliser pour le test
new_data = {
    'credit_lines_outstanding': 1,
    'loan_amt_outstanding': 1000.0,
    'total_debt_outstanding': 500.0,
    'income': 30000.0,
    'years_employed': 5,
    'fico_score': 600
}

def test_predict():
    # Effectuer une prédiction en appelant la fonction `model_pred`
    prediction = model_pred(new_data)
    
    # Vérifier que la prédiction est dans l'intervalle attendu
    assert prediction in [0, 1], "valeur de prédiction incorrecte"
    print("Test réussi : La prédiction est dans l'intervalle attendu.")

if __name__ == "__main__":
    test_predict()

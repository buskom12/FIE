def evaluate_prediction(predicted_prob, outcome):

    if outcome == 1:
        error = 1 - predicted_prob
    else:
        error = predicted_prob

    score = 1 - error

    return round(score, 3)

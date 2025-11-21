from fastapi import FastAPI, HTTPException
from typing import Optional
import pandas as pd
import numpy as np
import joblib
import uvicorn

# Initialize FastAPI app
app = FastAPI()

# Load the bundled models and encoders
def load_model_bundle():
    try:
        model_bundle = joblib.load('model.pkl')
        return model_bundle
    except FileNotFoundError:
        raise RuntimeError("model.pkl not found. Please ensure it's in the same directory.")

model_bundle = load_model_bundle()

# Extract components from the bundle
gbc_model = model_bundle['personalized_model']
visitorid_encoder = model_bundle['visitorid_encoder']
itemid_encoder = model_bundle['itemid_encoder']
event_encoder = model_bundle['event_encoder']
trending_items_df = model_bundle['trending_items']
visitor_item_map = model_bundle['visitor_item_map']

# Get the encoded index for 'transaction' event
transaction_event_index = np.where(event_encoder.classes_ == 'transaction')[0][0]

@app.get("/")
async def root():
    return {"message": "Welcome to the Recommendation API. Use /recommend/personalized/{visitor_id} for personalized recommendations or /recommend/trending for trending recommendations."}

@app.get("/recommend/personalized/{visitor_id}")
async def get_personalized_recommendations(visitor_id: int):
    """
    Generates top 5 personalized recommendations for a given visitor ID.
    """
    personalized_recs = []

    if visitor_id not in visitorid_encoder.classes_:
        return {"message": f"Visitor ID {visitor_id} not found in the dataset for personalized recommendations. No personalized recommendations available.", "recommendations": []}
    else:
        user_input_visitor_id_encoded = visitorid_encoder.transform([visitor_id])[0]

        interacted_items_encoded = visitor_item_map.get(user_input_visitor_id_encoded, [])

        all_items_encoded = itemid_encoder.transform(itemid_encoder.classes_)

        uninteracted_items_encoded = np.setdiff1d(all_items_encoded, interacted_items_encoded)

        if len(uninteracted_items_encoded) > 0:
            predict_df_personalized = pd.DataFrame({
                'visitorid': [user_input_visitor_id_encoded] * len(uninteracted_items_encoded),
                'itemid': uninteracted_items_encoded
            })

            transaction_probabilities = gbc_model.predict_proba(predict_df_personalized)[:, transaction_event_index]
            predict_df_personalized['transaction_probability'] = transaction_probabilities

            personalized_recommendations_df = predict_df_personalized.sort_values(
                by='transaction_probability', ascending=False
            ).head(5)

            personalized_recommendations_df['itemid_original'] = itemid_encoder.inverse_transform(
                personalized_recommendations_df['itemid']
            )
            personalized_recs = personalized_recommendations_df[['itemid_original', 'transaction_probability']].to_dict(orient='records')
        else:
            return {"message": f"Visitor {visitor_id} has interacted with all known items or no items are left to recommend.", "recommendations": []}

    return {"visitor_id": visitor_id, "recommendations": personalized_recs}

@app.get("/recommend/trending")
async def get_trending_recommendations():
    """
    Generates top 5 trending recommendations.
    """
    trending_recs = []

    # Trending items are already pre-calculated and stored in trending_items_df
    trending_recommendations_df = trending_items_df.sort_values(by='popularity_score', ascending=False).head(5).copy()
    trending_recommendations_df['itemid_original'] = itemid_encoder.inverse_transform(trending_recommendations_df['itemid'])
    trending_recs = trending_recommendations_df[['itemid_original', 'popularity_score']].to_dict(orient='records')

    return {"recommendations": trending_recs}

if __name__ == "__main__":
    # Run locally on 127.0.0.1:8000 (localhost)
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)

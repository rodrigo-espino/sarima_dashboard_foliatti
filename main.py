from flask import Flask, jsonify
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Neon database connection
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is not set")

@app.route("/api/sarima/<from_date>/<to_date>")
def sarima_forecast(from_date, to_date):
    # Validate date format
    try:
        datetime.strptime(from_date, '%Y-%m-%d')
        datetime.strptime(to_date, '%Y-%m-%d')
    except ValueError:
        return jsonify({'error': 'Invalid date format. Use YYYY-MM-DD'}), 400

    query = '''
        SELECT "DATE", AVG("Amount") as Amount
        FROM "Transactions"
        WHERE "DATE" >= %s AND "DATE" <= %s
        GROUP BY "DATE"
        ORDER BY "DATE" ASC
    '''
    
    try:
        # Create SQLAlchemy engine
        engine = create_engine(DATABASE_URL)
        
        # Execute query and load into DataFrame
        df = pd.read_sql(query, engine, params=(from_date, to_date))

        # Check if enough data is available
        if len(df) < 60:
            return jsonify({'msg': 'No hay suficientes datos para realizar la predicción'}), 400

        # Ensure 'DATE' is in datetime format
        df['DATE'] = pd.to_datetime(df['DATE'])
        df.set_index('DATE', inplace=True)

        # Fit SARIMA model
        sarima_model = SARIMAX(
            df['amount'],  # Fixed case sensitivity
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 30),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        sarima_result = sarima_model.fit(disp=False)

        # Forecast for 90 days
        forecast_sarima = sarima_result.get_forecast(steps=90)
        sarima_pred = forecast_sarima.predicted_mean

        # Generate future dates
        last_date = df.index.max()
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=90, freq='D')

        # Convert to JSON-serializable formats
        future_dates_str = [date.strftime('%Y-%m-%d') for date in future_dates]
        sarima_pred_values = sarima_pred.tolist()

        return jsonify({
            'date': future_dates_str,
            'y_pred': sarima_pred_values
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    finally:
        if 'engine' in locals():
            engine.dispose()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 8000)))
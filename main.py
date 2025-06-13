from flask import Flask, jsonify
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sqlalchemy import create_engine
import os

app = Flask(__name__)

# Neon database connection
DATABASE_URL = os.getenv("DATABASE_URL")  # Set in environment variables

@app.route("/api/sarima/<from_date>/<to_date>")
def sarima_forecast(from_date, to_date):
    query = '''
        SELECT "DATE", AVG("Amount") as Amount
        FROM "Transactions"
        WHERE "DATE" >= %s AND "DATE" <= %s
        GROUP BY "DATE"
        ORDER BY "DATE" ASC
    '''

    try:
        # Create SQLAlchemy engine
        e
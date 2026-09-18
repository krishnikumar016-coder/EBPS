"""
Flask application entry point for the Employee Burnout Prediction System.
Serves the REST API and static frontend dashboard.
"""

import os
import sys

# Add src directory to path
src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
sys.path.insert(0, src_dir)

from flask import Flask, send_from_directory
from flask_cors import CORS
from database import init_db
from api.routes import api


def create_app():
    """Application factory."""
    app = Flask(
        __name__,
        static_folder="static",
        static_url_path="/static",
    )

    # Enable CORS for all API routes
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    # Register API blueprint
    app.register_blueprint(api)

    # Initialize database
    with app.app_context():
        init_db()

    # Serve frontend
    @app.route("/")
    def index():
        return send_from_directory("static", "index.html")

    # Serve plot images from the plots directory
    @app.route("/plots/<path:filename>")
    def serve_plot(filename):
        plots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
        return send_from_directory(plots_dir, filename)

    # Error handlers
    @app.errorhandler(404)
    def not_found(e):
        return {"error": "Resource not found"}, 404

    @app.errorhandler(500)
    def internal_error(e):
        return {"error": "Internal server error"}, 500

    return app


if __name__ == "__main__":
    app = create_app()
    print("\n" + "=" * 55)
    print("  EMPLOYEE BURNOUT PREDICTION SYSTEM")
    print("  Dashboard: http://localhost:5000")
    print("  API Base:  http://localhost:5000/api")
    print("=" * 55 + "\n")
    app.run(debug=True, host="0.0.0.0", port=5000)

from flask import Flask
from web_app.routes import crop_routes  # crop blueprint
from web_app.routes.yield_routes import yield_routes  # yield blueprint
from web_app.routes.profit_routes import profit_pest_routes  # merged blueprint
from web_app.routes.pest_routes import pest_routes  # (keep only if you still want a separate pest module)
from web_app.routes.fertilizer_routes import fertilizer_routes
from web_app.routes.disease_routes import disease_bp
from web_app.routes.treatment_routes import treatment_routes


app = Flask(__name__)

# Register blueprints
app.register_blueprint(crop_routes.bp)
app.register_blueprint(yield_routes)
app.register_blueprint(profit_pest_routes)  
app.register_blueprint(pest_routes)  
app.register_blueprint(fertilizer_routes)
app.register_blueprint(disease_bp, url_prefix="/")
app.register_blueprint(treatment_routes)

if __name__ == "__main__":
    app.run(debug=True)

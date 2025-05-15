from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from os import path, environ
from flask_login import LoginManager

db = SQLAlchemy()
DB_NAME = "database.db"


def create_app():
    app = Flask(__name__)
    # Use environment variable for secret key with a fallback for development
    app.config['SECRET_KEY'] = environ.get('SECRET_KEY', 'dev-key-please-change-in-production')
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{DB_NAME}'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    db.init_app(app)

    from .views import views
    from .auth import auth

    app.register_blueprint(views, url_prefix='/')
    app.register_blueprint(auth, url_prefix='/')

    from .models import User, Note
    
    with app.app_context():
        try:
            db.create_all()
        except Exception as e:
            print(f"Error creating database: {e}")

    login_manager = LoginManager()
    login_manager.login_view = 'auth.login'
    login_manager.init_app(app)

    @login_manager.user_loader
    def load_user(id):
        try:
            return User.query.get(int(id))
        except:
            return None

    return app


def create_database(app):
    if not path.exists('website/' + DB_NAME):
        with app.app_context():
            db.create_all()
            print('Created Database!')

import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Flask
    SECRET_KEY = os.environ.get('SECRET_KEY', 'factoryiq-secret-key-2024')
    DEBUG = os.environ.get('DEBUG', 'True').lower() == 'true'

    # Database
    DATABASE_PATH = os.environ.get('DATABASE_PATH', 'factoryiq.db')

    # Groq
    GROQ_API_KEY = os.environ.get('GROQ_API_KEY', '')
    GROQ_MODEL = os.environ.get('GROQ_MODEL', 'llama3-8b-8192')

    # Application
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB
    UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', 'uploads')

    # Normal operating ranges for each parameter
    NORMAL_RANGES = {
        'temperature': {'min': 70.0, 'max': 85.0, 'unit': '°C'},
        'pressure':    {'min': 4.5,  'max': 6.5,  'unit': 'bar'},
        'vibration':   {'min': 0.1,  'max': 2.5,  'unit': 'mm/s'},
        'humidity':    {'min': 40.0, 'max': 65.0, 'unit': '%'},
        'speed':       {'min': 800,  'max': 1200, 'unit': 'RPM'},
        'production_rate': {'min': 85, 'max': 110, 'unit': 'units/hr'},
    }

    # Risk thresholds
    RISK_THRESHOLDS = {
        'quality_score': {'warning': 75, 'critical': 60},
        'defect_probability': {'warning': 0.3, 'critical': 0.6},
        'process_stability': {'warning': 70, 'critical': 50},
    }

    # Simulation settings
    SIMULATION_INTERVAL = 5  # seconds
    HISTORY_WINDOW = 100     # number of records to keep per machine for analysis

class DevelopmentConfig(Config):
    DEBUG = True

class ProductionConfig(Config):
    DEBUG = False

config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}

def get_config():
    env = os.environ.get('FLASK_ENV', 'default')
    return config.get(env, config['default'])

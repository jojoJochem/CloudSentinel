"""
Django settings for monitoring_project project.

Generated by 'django-admin startproject' using Django 5.0.3.

For more information on this file, see
https://docs.djangoproject.com/en/5.0/topics/settings/

For the full list of settings and their values, see
https://docs.djangoproject.com/en/5.0/ref/settings/
"""

from pathlib import Path
# import json
import os

# # Load  settings
# with open('monitoring_project/config.json') as config_file:
#     dynamic_config = json.load(config_file)
# globals().update(dynamic_config)
# ALLOWED_HOSTS = os.getenv('DJANGO_ALLOWED_HOSTS', 'localhost').split(',')

# # testing purposes
API_DATA_INGESTION_URL = "http://127.0.0.1:5001"
API_DATA_PROCESSING_URL = "http://127.0.0.1:5002"
API_CRCA_ANOMALY_DETECTION_URL = "http://127.0.0.1:5023"
API_CGNN_ANOMALY_DETECTION_URL = "http://127.0.0.1:5013"
API_LEARNING_ADAPTATION_URL = "http://127.0.0.1:5005"

# API_DATA_INGESTION_URL = os.getenv('API_DATA_INGESTION_URL', 'http://data-ingestion-service.cloudsentinel.svc.cluster.local:80')
# API_DATA_PROCESSING_URL = os.getenv('API_DATA_PROCESSING_URL', 'http://data-processing-service.cloudsentinel.svc.cluster.local:80')
# API_CRCA_ANOMALY_DETECTION_URL = os.getenv('API_CRCA_ANOMALY_DETECTION_URL', 'http://crca-anomaly-detection-service.cloudsentinel.svc.cluster.local:80')
# API_CGNN_ANOMALY_DETECTION_URL = os.getenv('API_CGNN_ANOMALY_DETECTION_URL', 'http://cgnn-anomaly-detection-service.cloudsentinel.svc.cluster.local:80')
# API_LEARNING_ADAPTATION_URL = os.getenv('API_LEARNING_ADAPTATION_URL', 'http://learning-adaptation-service.cloudsentinel.svc.cluster.local:80')

PROMETHEUS_URL = os.getenv('PROMETHEUS_URL', 'http://prometheus-server.monitoring.svc.cluster.local:80')
CLUSTER_NAMESPACE = os.getenv('CLUSTER_NAMESPACE', 'kube-system')

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/5.0/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = "django-insecure-&egy-$lq%)&cr4-c4=efe7*3eozqo(ej^9hcic7%^p3(()hboo"

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = False

ALLOWED_HOSTS = ['34.44.116.210', 'localhost', '127.0.0.1']


# Application definition

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "config_app",
    'crispy_forms',
    'rest_framework',
]

STATIC_URL = '/static/'

STATICFILES_DIRS = [
    BASE_DIR / "static",
]

STATIC_ROOT = os.path.join(BASE_DIR, 'staticfiles')

CRISPY_TEMPLATE_PACK = 'bootstrap4'

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'whitenoise.middleware.WhiteNoiseMiddleware',
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "monitoring_project.urls"

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = "monitoring_project.wsgi.application"


# Database
# https://docs.djangoproject.com/en/5.0/ref/settings/#databases

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}


# Password validation
# https://docs.djangoproject.com/en/5.0/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        "NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.MinimumLengthValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.CommonPasswordValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.NumericPasswordValidator",
    },
]


# Internationalization
# https://docs.djangoproject.com/en/5.0/topics/i18n/

LANGUAGE_CODE = "en-us"

TIME_ZONE = "UTC"

USE_I18N = True

USE_TZ = True


# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/5.0/howto/static-files/

STATIC_URL = "static/"

# Default primary key field type
# https://docs.djangoproject.com/en/5.0/ref/settings/#default-auto-field

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"


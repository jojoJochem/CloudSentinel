from django.shortcuts import render
import json


def home(request):
    with open('monitoring_project/config.json') as config_file:
        config_data = json.load(config_file)
    return render(request, 'config_app/home.html', {'config': config_data})

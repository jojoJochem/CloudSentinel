from django.shortcuts import render
from .utils import get_settings


def home(request):
    """
    Renders the home page

    Args:
        request (HttpRequest): The request object.

    Returns:
        HttpResponse: The rendered home page with configuration.
    """
    config_data = get_settings()
    return render(request, 'config_app/home.html', {'config': config_data})

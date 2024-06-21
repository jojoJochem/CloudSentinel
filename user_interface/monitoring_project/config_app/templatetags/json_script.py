import json
from django.utils.safestring import mark_safe
from django import template

register = template.Library()


@register.filter(name='json_script')
def json_script(value):
    return mark_safe(json.dumps(value, indent=2))
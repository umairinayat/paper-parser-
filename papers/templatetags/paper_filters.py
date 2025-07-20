from django import template

register = template.Library()

@register.filter
def split(value, arg):
    """
    Split a string by the given delimiter
    Usage: {{ value|split:"," }}
    """
    return value.split(arg)

@register.filter
def get_filename(value):
    """
    Get filename from file path
    Usage: {{ file.name|get_filename }}
    """
    return value.split('/')[-1] if value else '' 
from django.db import models


class RCAData(models.Model):
    metrics = models.TextField()
    image_data = models.TextField(null=True)  # Base64 encoded image
    created_at = models.DateTimeField(auto_now_add=True)

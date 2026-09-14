from django.db import models
import uuid

class DashboardShareToken(models.Model):
    token = models.UUIDField(default=uuid.uuid4, editable=False, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)
    expires_at = models.DateTimeField(db_index=True)
    # Storing state/parameters to recreate dashboard view
    state = models.JSONField(default=dict)
    
    @property
    def is_valid(self):
        from django.utils import timezone
        return timezone.now() < self.expires_at

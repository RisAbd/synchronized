from django.contrib import admin

from .models import Recitation


@admin.register(Recitation)
class RecitationAdmin(admin.ModelAdmin):
    list_display = ("id", "title", "reciter", "status", "stage", "created_at")
    list_filter = ("status", "source_type")
    search_fields = ("title", "title_ar", "reciter", "source_url")
    readonly_fields = ("created_at", "updated_at", "data")

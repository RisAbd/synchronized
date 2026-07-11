from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("add", views.add, name="add"),
    path("r/<int:pk>/run", views.run, name="run"),
    path("r/<int:pk>/delete", views.delete, name="delete"),
    path("r/<int:pk>/", views.player, name="player"),
    path("r/<int:pk>/data.json", views.data_json, name="data_json"),
    path("api/recitations", views.api_recitations, name="api_recitations"),
    path("r/<int:pk>/audio", views.audio, name="audio"),
    path("r/<int:pk>/status", views.status, name="status"),
]

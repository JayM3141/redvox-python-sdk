from django.urls import path
from . import views

urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    path('inspect/', views.inspect, name='inspect'),
    path('converter/', views.converter, name='converter'),
    path('validator/', views.validator, name='validator'),
    path('cli/', views.cli_runner, name='cli_runner'),
    path('analysis/', views.analysis, name='analysis'),
    path('cloud/', views.cloud, name='cloud'),
    path('samples/', views.samples, name='samples'),
    path('samples/download/<str:filename>/', views.download_sample, name='download_sample'),
    path('api/info/', views.api_info, name='api_info'),
]

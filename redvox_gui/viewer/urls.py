from django.urls import path, re_path
from rest_framework import permissions
from drf_yasg.views import get_schema_view
from drf_yasg import openapi
from . import views
from . import auth_views

schema_view = get_schema_view(
   openapi.Info(
      title="RedVox SDK GUI API",
      default_version='v1',
      description="API documentation for RedVox GUI",
   ),
   public=True,
   permission_classes=(permissions.AllowAny,),
)

urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    path('map/', views.map_dashboard, name='map_dashboard'),
    path('inspect/', views.inspect, name='inspect'),
    path('data_window/', views.data_window, name='data_window'),
    path('converter/', views.converter, name='converter'),
    path('validator/', views.validator, name='validator'),
    path('cli/', views.cli_runner, name='cli_runner'),
    path('analysis/', views.analysis, name='analysis'),
    path('cloud/', views.cloud, name='cloud'),
    path('samples/', views.samples, name='samples'),
    path('samples/download/<str:filename>/', views.download_sample, name='download_sample'),
    path('api/info/', views.api_info, name='api_info'),
    path('api/bulk_process/', views.bulk_process_directory, name='bulk_process_directory'),
    path('api/classify_sensors/', views.classify_sensors, name='classify_sensors'),
    path('api/generate_report/', views.generate_dashboard_report, name='generate_dashboard_report'),
    path('api/advanced_ml/', views.advanced_ml_classification, name='advanced_ml_classification'),
    path('api/generate_video/', views.generate_scientific_video, name='generate_scientific_video'),
    path('api/generate_3d/', views.generate_3d_visualization, name='generate_3d_visualization'),
    path('api/advanced_analytics/', views.advanced_analytics, name='advanced_analytics'),
    path('api/report/export/pdf/', views.export_pdf_report, name='export_pdf_report'),
    path('api/export/hdf5/', views.export_hdf5, name='export_hdf5'),
    path('api/export/netcdf/', views.export_netcdf, name='export_netcdf'),
    path('api/export/cloud/', views.export_cloud, name='export_cloud'),
    path('api/ml/analyze_audio/', views.api_ml_analyze_audio, name='api_ml_analyze_audio'),
    path('api/gis/filter/', views.gis_filter, name='gis_filter'),
    path('api/dashboard/share/', views.create_dashboard_share, name='create_dashboard_share'),
    path('shared/<uuid:token_id>/', views.view_shared_dashboard, name='view_shared_dashboard'),
    path('api/device_info/', views.get_device_info, name='get_device_info'),
    path('api/auth/login/', auth_views.api_login, name='api_login'),
    path('api/auth/logout/', auth_views.api_logout, name='api_logout'),
    path('api/auth/check/', auth_views.check_auth, name='check_auth'),
    re_path(r'^swagger(?P<format>\.json|\.yaml)$', schema_view.without_ui(cache_timeout=0), name='schema-json'),
    path('swagger/', schema_view.with_ui('swagger', cache_timeout=0), name='schema-swagger-ui'),
    path('redoc/', schema_view.with_ui('redoc', cache_timeout=0), name='schema-redoc'),
]

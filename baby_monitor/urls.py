from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('audio_analysis.urls')),
    path('health/', include('audio_analysis.urls')),
]
# audio_analysis/urls.py
from django.urls import path
from . import views
from .views import UserListCreateView

urlpatterns = [
    # Original endpoints
    path('analyze-audio/', views.analyze_audio, name='analyze_audio'),
    path('health/', views.health_check, name='health_check'),
    path('users/', UserListCreateView.as_view(), name='user-list-create'),
    
    # New prediction endpoints
    path('predict-audio/', views.predict_audio, name='predict_audio'),
    path('test-prediction/', views.test_audio_prediction, name='test_prediction'),
]
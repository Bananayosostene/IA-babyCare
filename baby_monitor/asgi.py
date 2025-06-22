import os
from django.core.asgi import get_asgi_application
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack
from channels.security.websocket import AllowedHostsOriginValidator
import audio_analysis.routing

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'baby_monitor.settings')

application = ProtocolTypeRouter({
    "http": get_asgi_application(),
    "websocket": AllowedHostsOriginValidator(
        AuthMiddlewareStack(
            URLRouter(
                audio_analysis.routing.websocket_urlpatterns
            )
        )
    ),
})

from rest_framework.authentication import BaseAuthentication
from rest_framework.exceptions import AuthenticationFailed
from .models import User

class TokenAuthentication(BaseAuthentication):
    def authenticate(self, request):
        auth_header = request.META.get('HTTP_AUTHORIZATION')
        if not auth_header or not auth_header.startswith('Bearer '):
            return None
        
        token = auth_header.split(' ')[1]
        try:
            # In a real app, you'd validate the JWT token here
            # For now, we'll just use the user_id as the token
            user = User.objects.get(id=token)
            return (user, token)
        except User.DoesNotExist:
            raise AuthenticationFailed('Invalid token')
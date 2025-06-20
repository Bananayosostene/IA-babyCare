from django.db import models
import uuid

class BabyState(models.TextChoices):
    CRYING = 'crying', 'Crying'
    LAUGHING = 'laughing', 'Laughing'
    CALM = 'calm', 'Calm'
    NEUTRAL = 'neutral', 'Neutral'

class AudioSessionStatus(models.TextChoices):
    ACTIVE = 'active', 'Active'
    ENDED = 'ended', 'Ended'
    ERROR = 'error', 'Error'

class DeviceStatus(models.TextChoices):
    ACTIVE = 'active', 'Active'
    INACTIVE = 'inactive', 'Inactive'
    MAINTENANCE = 'maintenance', 'Maintenance'

class DeviceType(models.TextChoices):
    SENSOR = 'sensor', 'Sensor'
    SPEAKER = 'speaker', 'Speaker'
    CAMERA = 'camera', 'Camera'

class User(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    username = models.CharField(max_length=150)
    email = models.EmailField(unique=True)
    password = models.CharField(max_length=128)
    phone = models.CharField(max_length=20, null=True, blank=True)
    location = models.CharField(max_length=255, null=True, blank=True)
    role = models.CharField(max_length=10, default='BABY')
    status = models.CharField(max_length=10, default='active')
    createdAt = models.DateTimeField(auto_now_add=True)
    updatedAt = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'User'

class Baby(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=100)
    birth_date = models.DateTimeField()
    age = models.IntegerField()
    parent = models.ForeignKey(User, on_delete=models.CASCADE, related_name='babies')
    state = models.CharField(max_length=10, choices=BabyState.choices, default=BabyState.CALM)
    selected_audio_id = models.UUIDField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'Baby'

class Device(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=100)
    type = models.CharField(max_length=10, choices=DeviceType.choices)
    baby = models.ForeignKey(Baby, on_delete=models.CASCADE, related_name='devices')
    status = models.CharField(max_length=15, choices=DeviceStatus.choices, default=DeviceStatus.ACTIVE)
    serial_number = models.CharField(max_length=100, unique=True)
    last_ping = models.DateTimeField(auto_now=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'Device'

class AudioSession(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    baby = models.ForeignKey(Baby, on_delete=models.CASCADE, related_name='audio_sessions')
    start_time = models.DateTimeField(auto_now_add=True)
    end_time = models.DateTimeField(null=True, blank=True)
    total_chunks = models.IntegerField(default=0)
    status = models.CharField(max_length=10, choices=AudioSessionStatus.choices, default=AudioSessionStatus.ACTIVE)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'AudioSession'

class SensorData(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    device_id = models.UUIDField()
    baby = models.ForeignKey(Baby, on_delete=models.CASCADE, related_name='sensor_data')
    status = models.CharField(max_length=10, choices=BabyState.choices)
    timestamp = models.DateTimeField(auto_now_add=True)
    confidence_score = models.FloatField(null=True, blank=True)
    audio_session = models.ForeignKey(AudioSession, on_delete=models.CASCADE, null=True, blank=True)

    class Meta:
        db_table = 'SensorData'
        indexes = [
            models.Index(fields=['baby', 'timestamp']),
            models.Index(fields=['device_id', 'timestamp']),
            models.Index(fields=['audio_session']),
            models.Index(fields=['confidence_score']),
        ]

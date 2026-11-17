from django.db import models
from django.utils.timezone import now  # For default timestamp

class FileUpload(models.Model):
    file = models.FileField(upload_to='uploads/')
    uploaded_at = models.DateTimeField(default=now)
    result = models.TextField(blank=True, null=True)  # Store relative path to processed file

    def __str__(self):
        return f"{self.file.name} - {self.uploaded_at.strftime('%Y-%m-%d %H:%M:%S')}"

class Contact(models.Model):
    # Fields for the Contact model
    name = models.CharField(max_length=255)  # Increased max_length for flexibility
    email = models.EmailField()  # Changed to EmailField for validation
    phone = models.CharField(max_length=15)  # Standard phone number length
    message = models.TextField()  # Renamed to 'message' for better readability
    submitted_at = models.DateTimeField(default=now)  # Automatically set current datetime

    def __str__(self):
        # Show name and submission date in admin
        return f"{self.name} - {self.submitted_at.strftime('%Y-%m-%d %H:%M:%S')}"

    class Meta:
        # Additional metadata options
        ordering = ['-submitted_at']  # Show recent submissions first
        verbose_name = "Contact Submission"  # Admin panel display name
        verbose_name_plural = "Contact Submissions"

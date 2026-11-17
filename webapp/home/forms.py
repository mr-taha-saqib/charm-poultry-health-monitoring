# forms.py
from django import forms

class UploadForm(forms.Form):
    file = forms.FileField(label='Select an image or video', required=True)

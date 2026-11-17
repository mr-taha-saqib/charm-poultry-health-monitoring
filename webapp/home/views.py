import os
import posixpath

from django.conf import settings
from django.contrib import messages
from django.core.files.storage import FileSystemStorage
from django.http import HttpResponseBadRequest
from django.shortcuts import redirect, render
from django.utils.timezone import now

from .detection import run_detection_and_tracking
from .forms import UploadForm
from .models import Contact, FileUpload


def index(request):
    """Renders the homepage."""
    return render(request, "index.html")

def about(request):
    """Renders the about page."""
    return render(request, "about.html")

def live(request):
    """Renders the live page."""
    return render(request, "Live.html")

def contact(request):
    """Handles contact form submissions and saves them to the database."""
    if request.method == "POST":
        name = request.POST.get('name')
        email = request.POST.get('email')
        phone = request.POST.get('phone')
        message = request.POST.get('message')

        try:
            contact = Contact(
                name=name,
                email=email,
                phone=phone,
                message=message,
                submitted_at=now()
            )
            contact.save()
            messages.success(request, "Your message has been sent successfully!")
        except Exception as e:
            messages.error(request, f"Error saving contact form: {str(e)}")
    
    return render(request, "contact.html")

def upload_file(request):
    if request.method == 'POST':
        form = UploadForm(request.POST, request.FILES)
        
        if form.is_valid():
            uploaded_file = request.FILES['file']
            
            if uploaded_file.size > settings.MAX_UPLOAD_SIZE:
                return HttpResponseBadRequest("File too large.")
            
            fs = FileSystemStorage()
            filename = fs.save(uploaded_file.name, uploaded_file)
            file_path = fs.path(filename)
            
            output_dir = os.path.join(settings.MEDIA_ROOT, 'processed')
            os.makedirs(output_dir, exist_ok=True)
            
            try:
                processed_file_path = run_detection_and_tracking(file_path, output_dir)
                if not processed_file_path or not os.path.exists(processed_file_path):
                    raise Exception("Processed file not created.")
                
                relative_processed_path = os.path.relpath(processed_file_path, settings.MEDIA_ROOT)
                
                file_upload = FileUpload(file=filename, result=relative_processed_path)
                file_upload.save()
                
                messages.success(request, "File uploaded and processed successfully!")
                return redirect('result', file_id=file_upload.id)
            
            except Exception as e:
                messages.error(request, f"Error during file processing: {str(e)}")
                return redirect('upload')
        else:
            messages.error(request, "Invalid form submission.")
    
    else:
        form = UploadForm()
    
    return render(request, 'upload.html', {'form': form})


def result(request, file_id):
    """Display uploaded file and processed result."""
    try:
        file_upload = FileUpload.objects.get(id=file_id)

        # Both paths are stored relative to MEDIA_ROOT: the upload is saved at
        # its root and the processed output under processed/.
        result_path = (file_upload.result or "").replace(os.sep, "/")

        context = {
            'fileupload': file_upload,
            'original_url': posixpath.join(settings.MEDIA_URL, file_upload.file.name),
            'result_url': posixpath.join(settings.MEDIA_URL, result_path) if result_path else None,
        }

        return render(request, 'result.html', context)

    except FileUpload.DoesNotExist:
        messages.error(request, "File not found.")
        return redirect('upload')
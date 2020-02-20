from django.shortcuts import render, redirect
from .forms import UploadForm, ImageUploadForm
from django.core.files.storage import FileSystemStorage
from .cv_functions import cv_detect_number
from django.conf import settings

# Create your views here.
def first_view(request):
    return render(request, 'mnist/first_view.html', {})

def upload(request):
    if request.method == 'POST':
        form = UploadForm(request.POST, request.FILES)

        if form.is_valid():
            myfile = request.FILES['image']
            fs = FileSystemStorage()
            filename = fs.save(myfile.name, myfile)
            uploaded_file_url = fs.url(filename)

            context = {'form':form, 'uploaded_file_url':uploaded_file_url}
            return render(request, 'mnist/upload.html', context)
    else:
        form = UploadForm()
        context = {'form':form}
        return render(request, 'mnist/upload.html', context)

def detect_number(request):
    if request.method == 'POST':

        form = ImageUploadForm(request.POST, request.FILES)

        if form.is_valid():

            post = form.save(commit=False)
            post.save()
            imageURL = settings.MEDIA_URL + form.instance.document.name
            # print(form.instance, form.instance.document.name, form.instance.document.url)
            result = cv_detect_number(settings.MEDIA_ROOT_URL + imageURL)
            return render(request, 'mnist/detect_number.html',{'form':form, 'post':post,'result':result})
            #return render(request, 'opencv_webapp/template/index.html',{'form':form, 'post':post,'result':result})
    else:
        form = ImageUploadForm()
        return render(request, 'mnist/detect_number.html', {'form':form})
        #return render(request, 'opencv_webapp/template/index.html', {'form':form})

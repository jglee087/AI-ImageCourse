from django import forms
from .models import ImageUploadModel

# 1) Hard-coded HTML : form, input, label, button, ....
# 2) forms.Form : 양식(From class) 안에 받아들일 Field 선언
# 3) forms.ModelForm : Model class 안에 받아들일 Field를 선언하고 해동 클래스를 당겨와 활용

class UploadForm(forms.Form):
    title = forms.CharField(max_length=50)
    image = forms.ImageField()

class ImageUploadForm(forms.ModelForm):
    class Meta:
        model = ImageUploadModel
        fields = ('description', 'document',)

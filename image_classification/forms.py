from django import forms

class ImageUpLoadForm(forms.Form): 
    image = forms.ImageField()
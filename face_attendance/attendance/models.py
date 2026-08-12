from django.db import models

class Student(models.Model):
    name = models.CharField(max_length=100, unique=True) 
    roll_no = models.CharField(max_length=20, unique=True) 
    image = models.ImageField(upload_to='faces/') 

    def __str__(self):
        return f"{self.roll_no} - {self.name}"
from django.db import models
from django.contrib.auth.models import AbstractUser
from My_project import settings

class BaseClass(models.Model):#abstract class for adding create and modified time details
    created_at = models.DateTimeField(auto_now_add=True)
    modified_at = models.DateTimeField(auto_now=True)

    class Meta:
        abstract = True

class User(AbstractUser,BaseClass):# class to access existing user class and add needed modifications
    email = models.EmailField(max_length=50, unique=True)
    mobile = models.IntegerField(max_length=10)
    REQUIRED_FIELDS = ['email','mobile']
        
    class Meta:
        db_table = 'auth_user'
        verbose_name = 'user'
        verbose_name_plural = 'users'

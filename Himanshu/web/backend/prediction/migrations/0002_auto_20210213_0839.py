# Generated by Django 3.1.6 on 2021-02-13 08:39

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('prediction', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='image',
            name='uploads',
            field=models.ImageField(upload_to='uploaded_images/'),
        ),
    ]

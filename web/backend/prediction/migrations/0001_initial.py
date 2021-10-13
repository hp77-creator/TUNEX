# Generated by Django 3.1.6 on 2021-02-06 15:33

from django.db import migrations, models
import django_resized.forms
import prediction.models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Image',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('category', models.CharField(default='Upload Image', max_length=256)),
                ('uploads', django_resized.forms.ResizedImageField(crop=None, force_format=None, keep_meta=True, quality=0, size=[200, 200], storage=prediction.models.OverwriteStorage(), upload_to='uploaded_images/')),
            ],
        ),
    ]
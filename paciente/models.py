from django.db import models
from django.contrib.auth.models import User
import hashlib
from django.core.validators import FileExtensionValidator
import os

# Clase Paciente que representa los datos de un paciente en la base de datos

class Paciente(models.Model):
    id_paciente = models.AutoField(primary_key=True)
    user = models.ForeignKey(User, on_delete=models.CASCADE)  # Relación con el modelo User
    nombre_paciente = models.CharField(max_length=45, verbose_name= 'Nombre del paciente', null=False, blank=True)
    apellido_paterno = models.CharField(max_length=45, verbose_name= 'Apellido paterno', null=False, blank=True)
    apellido_materno = models.CharField(max_length=45, verbose_name= 'Apellido materno', null=False, blank=True)
    telefono = models.CharField(max_length=15, verbose_name= 'Telefono', null=False, blank=True)
    correo = models.CharField(max_length=45, verbose_name= 'Correo electrónico', null=False, blank=True)
    sexo = models.CharField(max_length=15, verbose_name= 'Sexo', null=False, blank=True)
    fecha_nacimiento = models.DateField(verbose_name= 'Fecha Nacimiento', null=False, blank=True)
    imc = models.FloatField()
    uso_de_medicamentos = models.CharField(max_length=100, null=True, blank=True, verbose_name= 'Medicamentos')  # Nuevo campo
    actividad_fisica = models.CharField(max_length=100, null=True, blank=True, verbose_name= 'Actividad física')      # Nuevo campo
    foto = models.ImageField(upload_to=' ', verbose_name= 'Fotografía del paciente', null=True)
    
    # Relación con Especialista: cada paciente está vinculado a un especialista específico
    especialista = models.ForeignKey('Especialista', on_delete=models.CASCADE, related_name="pacientes")

    def __str__(self):
        return f"{self.nombre_paciente} {self.apellido_paterno} - {self.especialista.user.username}"



class Departamento(models.Model):
    id_departamento = models.AutoField(primary_key=True)
    departamento = models.CharField(max_length=45, null=False, blank=True)

    def __str__(self):
        return self.departamento



class Especialista(models.Model):
    id_especialista = models.AutoField(primary_key=True)
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    nombre_especialista = models.CharField(max_length=100)
    apellido_paterno = models.CharField(max_length=100)
    apellido_materno = models.CharField(max_length=100)
    telefono = models.CharField(max_length=15)
    correo = models.EmailField(max_length=100)
    especialidad = models.CharField(max_length=100)
    fecha_nacimiento = models.DateField(verbose_name= 'Fecha Nacimiento', null=False, blank=True)
    departamento_id = models.ForeignKey(Departamento, on_delete=models.CASCADE)  # Cambiado a ForeignKey

    def __str__(self):
        return (f"{self.id_especialista} - {self.nombre_especialista} {self.apellido_paterno} {self.apellido_materno}, "
                f"Fecha_Naciemiento: {self.fecha_nacimiento}, Teléfono: {self.telefono}, Correo: {self.correo}, "
                f"Especialidad: {self.especialidad}, Departamento: {self.departamento_id}")

        
 
class ECG(models.Model):
    id_ecg = models.AutoField(primary_key=True)
    archivo_ecg = models.FileField(upload_to= 'archivo_ecg/', validators=[FileExtensionValidator(allowed_extensions=['txt', 'csv'])])
    fecha_informe = models.DateTimeField(auto_now_add=True)
    comentarios = models.TextField()
    paciente = models.ForeignKey(Paciente, on_delete=models.CASCADE, db_column='ID_PACIENTE')
    homoclave = models.CharField(max_length=64, unique=True, blank=True, null=True)  # Nuevo campo para homoclave

    def save(self, *args, **kwargs):
        
        if not self.homoclave:  # Generar homoclave solo si no existe
            salt = os.urandom(16)
            # Usar un hash SHA-256 para generar la homoclave
            fullhash = hashlib.sha256(salt).hexdigest()
            self.homoclave = fullhash[:10].upper()
        #sobreescritura de métodos
        super().save(*args, **kwargs)  # Llama al método save original

   
    def __str__(self):
        return f'ECG paciente: {self.paciente.nombre_paciente} {self.paciente.apellido_paterno} {self.paciente.apellido_materno}'
 
       
class AnalisisDominioFrecuencia(models.Model):
    id_analisis_frecuencia = models.AutoField(primary_key=True)
    tasa_muestreo = models.IntegerField()
    power_high_frequency = models.FloatField()
    power_low_frequency = models.FloatField()
    potencia_total = models.FloatField()
    lh_hf = models.FloatField()
    espectro_frecuencias = models.ImageField(upload_to='media/', null = True)
    psd = models.ImageField(upload_to='media/', null=True)
    paciente = models.ForeignKey(Paciente, on_delete=models.CASCADE, db_column= 'id_paciente', null=False)
    fecha_creacion = models.DateTimeField(auto_now_add=True, null=True)
    ecg = models.ForeignKey(ECG, on_delete=models.CASCADE, db_column='ID_ECG')
    
    def __str__(self):
        return f'AF_{self.paciente.apellido_paterno[0]}{self.paciente.apellido_materno[0]}{self.paciente.nombre_paciente[0]}_{self.ecg.fecha_informe.strftime("%Y%m%d_%H:%M:%S")}'

    def delete(self, using = None, keep_parents = False):
        if self.psd:
            self.psd.delete(save=False)

        if self.espectro_frecuencias:
            self.espectro_frecuencias.delete(save=False)

        return super().delete(using=using, keep_parents=keep_parents)

class AnalisisDominioTiempo(models.Model):
    id_analisis_tiempo = models.AutoField(primary_key=True)
    nni_mean = models.FloatField()
    nni_min = models.FloatField()
    nni_max = models.FloatField()
    hr_mean = models.FloatField()
    hr_min = models.FloatField()
    hr_max = models.FloatField()
    std_hr = models.FloatField()
    nni_diff_mean = models.FloatField()
    nni_diff_min = models.FloatField()
    nni_diff_max = models.FloatField()
    std_rr = models.FloatField()
    ssdn_index = models.FloatField()
    sdann = models.FloatField()
    rmssd = models.FloatField()
    sdsd = models.FloatField(default=0.0)
    nn50 = models.FloatField()
    pnns50 = models.FloatField()
    nn20 = models.FloatField()
    rr_mean = models.FloatField()
    total_intervalos_rr = models.IntegerField()
    histograma_rr = models.ImageField(upload_to='media/', null=True )
    histograma_hr = models.ImageField(upload_to='media/', null=True)
    histograma_nni = models.ImageField(upload_to='media/', null=True)
    paciente = models.ForeignKey(Paciente, on_delete=models.CASCADE, db_column='id_paciente', null=False)
    fecha_creacion = models.DateTimeField(auto_now_add=True, null =True)

    
    def __str__(self):
        return f'AT_{self.paciente.apellido_paterno[0]}{self.paciente.apellido_materno[0]}{self.paciente.nombre_paciente[0]}_{self.fecha_creacion.strftime("%Y%m%d_%H:%M:%S")}'
    
    def delete(self, using = None, keep_parents = False):
        if self.histograma_rr:
            self.histograma_rr.delete(save=False)
        if self.histograma_hr:
            self.histograma_hr.delete(save=False)
        if self.histograma_nni:
            self.histograma_nni.delete(save=False)
        #mantener la herencia de datos 
        super().delete(using=using, keep_parents=keep_parents)
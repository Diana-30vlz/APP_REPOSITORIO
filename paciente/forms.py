from django import forms
from .models import Paciente, Departamento, Especialista, AnalisisDominioFrecuencia, AnalisisDominioTiempo, ECG
from django.contrib.auth.models import User

class PacienteForm(forms.ModelForm):
    class Meta:
        model = Paciente
        fields = [
            'nombre_paciente',
            'apellido_paterno',
            'apellido_materno',
            'telefono',
            'correo',
            'sexo',
            'fecha_nacimiento',
            'uso_de_medicamentos',
            'actividad_fisica',
            'imc'
        ]
        widgets = {
            'fecha_nacimiento': forms.DateInput(attrs={'type': 'date'}),
        }

class DepartamentoForm(forms.ModelForm):
    class Meta:
        model = Departamento
        fields = ['departamento']

class EspecialistaForm(forms.ModelForm):
    class Meta:
        model = Especialista
        fields = [
            'user',
            'nombre_especialista',
            'apellido_paterno',
            'apellido_materno',
            'telefono',
            'correo',
            'especialidad',
            'fecha_nacimiento',
            'departamento_id',
        ]
        widgets = {
            'fecha_nacimiento': forms.DateInput(attrs={'type': 'date'}),
        }

class AnalisisDominioFrecuenciaForm(forms.ModelForm):
    class Meta:
        model = AnalisisDominioFrecuencia
        fields = [
            'tasa_muestreo',
            'power_high_frequency',
            'power_low_frequency',  
            'potencia_total',
            'lh_hf',
            'espectro_frecuencias',
            'psd',
            'paciente',
        ]

class AnalisisDominioTiempoForm(forms.ModelForm):
    class Meta:
        model = AnalisisDominioTiempo
        fields = [
            'nni_mean',
            'nni_min',
            'nni_max',
            'hr_mean',
            'hr_min',
            'hr_max',
            'std_hr',
            'nni_diff_mean',
            'nni_diff_min',
            'nni_diff_max',
            'std_rr',
            'ssdn_index',
            'sdann',
            'rmssd',
            'sdsd',
            'nn50',
            'pnns50',
            'nn20',
            'rr_mean',
            'total_intervalos_rr',
            'histograma_rr',
            'histograma_hr',
            'histograma_nni',
            'paciente',
        ]

class ECGForm(forms.ModelForm):
    class Meta:
        model = ECG
        fields = [
            'archivo_ecg',
            'comentarios',
        ]
        widgets = {
            'fecha_informe': forms.DateInput(attrs={'type': 'date'}),
        }





class UserRegistrationForm(forms.ModelForm):
    password1 = forms.CharField(label='Contraseña', widget=forms.PasswordInput)
    password2 = forms.CharField(label='Confirmar contraseña', widget=forms.PasswordInput)

    class Meta:
        model = User
        fields = ['username', 'email','password1']  # Incluye los campos que deseas en tu formulario

    def clean_password2(self):
        password1 = self.cleaned_data.get("password1")
        password2 = self.cleaned_data.get("password2")

        if password1 and password2 and password1 != password2:
            raise forms.ValidationError("Las contraseñas no coinciden")
        return password2

    def save(self, commit=True):
        user = super(UserRegistrationForm, self).save(commit=False)
        user.set_password(self.cleaned_data["password1"])
        if commit:
            user.save()
        return user

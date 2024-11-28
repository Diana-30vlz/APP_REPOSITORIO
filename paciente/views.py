from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.models import User
from django.db import IntegrityError
from django.utils import timezone
from django.contrib.auth.decorators import login_required
from .models import *
from .forms import *
from django.db.models import Q
import pandas as pd 
from datetime import datetime


from weasyprint import HTML
from django.http import HttpResponse
from django.conf import settings

from django.template.loader import render_to_string

import numpy as np
from scipy.signal import firwin, lfilter, find_peaks, welch
from scipy.interpolate import interp1d
from pyhrv import time_domain
import plotly.express as px
import plotly.io as pio
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import json
import plotly
from django.core.serializers.json import DjangoJSONEncoder
from biosppy.signals import ecg
from django.utils import timezone







def signup(request):
    if request.method == 'GET':
        departamentos = Departamento.objects.all()
        form = UserRegistrationForm()
        return render(request, 'signup.html', {"form": form, "departamentos": departamentos})
    else:
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            password1 = form.cleaned_data.get('password1')
            password2 = form.cleaned_data.get('password2')
            if password1 == password2:
                try:
                    # Intento de crear el usuario
                    user = form.save(commit=False)
                    user.set_password(password1)  # Encripta la contraseña antes de guardar el usuario
                    user.save()

                    # Recoge los datos del formulario adicionales
                    nombre_especialista = request.POST.get("nombres")
                    apellido_paterno = request.POST.get("apellido_paterno")
                    apellido_materno = request.POST.get("apellido_materno")
                    telefono = request.POST.get("telefono")
                    correo = request.POST.get("correo")
                    especialidad = request.POST.get("especialidad")
                    departamento_id = request.POST.get("departamento")
                    fecha_nacimiento = request.POST.get("fecha_nacimiento")

                    # Validación de que la fecha de nacimiento no sea nula
                    if not fecha_nacimiento:
                        return render(request, 'signup.html', {
                            "form": form,
                            "departamentos": Departamento.objects.all(),
                            "error": "Por favor, proporciona una fecha de nacimiento válida."
                        })

                    # Obtiene la instancia de Departamento usando el ID proporcionado
                    departamento = Departamento.objects.get(id_departamento=departamento_id)

                    # Crea el objeto Especialista
                    especialista = Especialista.objects.create(
                        user=user,
                        nombre_especialista=nombre_especialista,
                        apellido_paterno=apellido_paterno,
                        apellido_materno=apellido_materno,
                        telefono=telefono,
                        correo=correo,
                        especialidad=especialidad,
                        fecha_nacimiento=fecha_nacimiento,
                        departamento_id=departamento  # Ahora asigna la instancia de Departamento
                    )
                    especialista.save()

                    login(request, user)
                    return redirect('homeDoctor')
                except IntegrityError as e:
                    print(e)  # Esto imprimirá la excepción completa en la consola.
                    if 'UNIQUE constraint' in str(e):
                        error_message = "El usuario ya existe. Prueba con otro nombre de usuario."
                    else:
                        error_message = f"Ocurrió un error durante el registro: {e}."  # Muestra el error específico
                    return render(request, 'signup.html', {
                        "form": form,
                        "departamentos": Departamento.objects.all(),
                        "error": error_message
                    })

            else:
                # Si las contraseñas no coinciden
                return render(request, 'signup.html', {
                    "form": form,
                    "departamentos": Departamento.objects.all(),
                    "error": "Las contraseñas no coinciden."
                })
        
        # Si el formulario no es válido
        return render(request, 'signup.html', {
            "form": form,
            "departamentos": Departamento.objects.all(),
            "error": "Por favor corrige los errores del formulario."
        })

# Vista para mostrar los pacientes pendientes (no completados)
@login_required
def pacientes(request):
    query = request.GET.get('query', '')  # Captura el parámetro de búsqueda desde la URL
    especialista = request.user.especialista  # Obtiene el especialista asociado al usuario actual

    if query:
        # Filtra pacientes del especialista actual que coincidan con el criterio de búsqueda
        pacientes = Paciente.objects.filter(
            Q(especialista=especialista) &
            (
                Q(nombre_paciente__icontains=query) |
                Q(apellido_paterno__icontains=query) |
                Q(apellido_materno__icontains=query) |
                Q(id_paciente__icontains=query) |
                Q(sexo__icontains=query) |
                Q(correo__icontains=query)
            )
        )
    else:
        # Si no hay búsqueda, muestra todos los pacientes del especialista actual
        pacientes = Paciente.objects.filter(especialista=especialista)

    return render(request, 'paciente.html', {"pacientes": pacientes})
    


@login_required
def editar_paciente(request, paciente_id):
    paciente = get_object_or_404(Paciente, id_paciente=paciente_id)  # Cambiado a id_paciente
    if request.method == 'POST':
        form = PacienteForm(request.POST, instance=paciente)
        if form.is_valid():
            form.save()
            return redirect('pacientes')
    else:
        form = PacienteForm(instance=paciente)
    return render(request, 'editar_paciente.html', {'form': form})

'''En la plantilla editar.html, puedes usar el formulario de la siguiente manera:
#{'Formulario': Formulario} significa que dentro de la plantillaeditar.html, puedes acceder a la instancia del formulario con la variable Formulario.'''


@login_required
def eliminar_paciente(request, paciente_id):
    pacientes = get_object_or_404(Paciente, id_paciente=paciente_id) # referencia al campo de la clase 
    pacientes.delete() # Elimina el paciente
    return redirect('pacientes') # Redirige a la lista de pacientes


@login_required
def historial(request, paciente_id):
    paciente = get_object_or_404(Paciente, id_paciente=paciente_id)
    registros_ecg = ECG.objects.filter(paciente=paciente)  # Filtrar por el paciente específico
    
    # Verificación de si hay registros
    no_registros = registros_ecg.count() == 0
    
    # Depuración: mostrar en la consola si no hay registros
    print(f"No hay registros para el paciente {paciente_id}: {no_registros}")
    
    return render(request, 'historial.html', {'paciente': paciente, 'registros_ecg': registros_ecg, 'no_registros': no_registros})


@login_required
def buscar(request):
    query = request.GET.get('query', '')
    pacientes = Paciente.objects.filter(
        Q(nombre_paciente__icontains=query) |
        Q(apellido_paterno__icontains=query)
    ) if query else []
    
    return render(request, 'paciente.html', {'pacientes': pacientes, 'query': query})

# Vista para crear un nuevo paciente
@login_required
def create_paciente(request):
    if request.method == "GET":
        return render(request, 'create_paciente.html', {"form": PacienteForm()})
    else:
        form = PacienteForm(request.POST)
        if form.is_valid():
            try:
                # Buscar el especialista vinculado al usuario actual
                especialista = Especialista.objects.get(user=request.user)
                
                new_paciente = form.save(commit=False)
                new_paciente.user = request.user  # Vincula el paciente con el usuario actual
                new_paciente.especialista_id = especialista.id_especialista  # Solo asigna el ID del especialista
                new_paciente.save()
                return redirect('pacientes')
            except Especialista.DoesNotExist:
                return render(request, 'create_paciente.html', {
                    "form": form,
                    "error": "El especialista no está registrado. Por favor, verifica tu cuenta."
                })
            except ValueError:
                return render(request, 'create_paciente.html', {
                    "form": form,
                    "error": "Surgió un error al crear al paciente."
                })
        print(form.errors)
        return render(request, 'create_paciente.html', {
            "form": form,
            "error": "Por favor corrige los errores del formulario."
        })

@login_required
def crear_informe(request, paciente_id):
    paciente = get_object_or_404(Paciente, id_paciente=paciente_id)
    Formulario = ExpedienteForm(request.POST or None, request.FILES or None)  # Cargar los datos del formulario

    if request.method == 'POST':  # Solo manejamos el archivo cuando sea POST
        if Formulario.is_valid():
            # Validación adicional para el archivo
            archivo = request.FILES.get('archivo_ecg')
            if archivo:
                if not (archivo.name.endswith('.txt') or archivo.name.endswith('.csv')):
                    Formulario.add_error('archivo_ecg', 'El archivo debe ser de tipo .txt o .csv.')
                else:
                    expediente = Formulario.save(commit=False)  # No guarda todavía
                    expediente.paciente = paciente  # Relaciona el paciente
                    expediente.save()  # Ahora guarda el informe

                    # Leer el archivo solo si es válido
                    try:
                        ECG = pd.read_csv(archivo, sep='\s+', header=None)
                        if ECG.shape[1]==2:
                            tiempo = ECG.iloc[:,0]
                            voltaje = ECG.iloc[:,1]
                            fm = int(1/(tiempo.iloc[1]-tiempo.iloc[0]))
                            distacia = fm * 0.2


                           # fig = px.line(x=tiempo, y=voltaje, labels={'x': 'Tiempo (s)', 'y': 'Voltaje (mV)'}, title='ECG')
                            
                            #fig_path = f'media/graficos/ecg_{paciente.id_paciente}.html'
                            #fig.write_html(fig_path)
                    except Exception as e:
                        Formulario.add_error('archivo_ecg', f'Error al leer el archivo: {str(e)}')
                    return redirect('historial', paciente_id=paciente.id_paciente)
            else:
                Formulario.add_error('archivo_ecg', 'Debes seleccionar un archivo.')
        else:
            print(Formulario.errors)  # Mostrar los errores si los hay

    # Si es una solicitud GET o si hay errores en el formulario, renderizamos de nuevo el formulario
    return render(request, 'crear_informe.html', {'Formulario': Formulario, 'id_paciente': paciente.id_paciente})

@login_required
def eliminar_informe(request, paciente_id):
    ecg = get_object_or_404(ECG, id_ecg=paciente_id) # referencia al campo de la clase 
    ecg.delete() # Elimina el paciente
    return redirect('historial', paciente_id=ecg.paciente.id_paciente ) # Redirige a la lista de pacientes








def formato_ecg(file):
    delimitadores = [",", "\t", " ", ";"]
    for delim in delimitadores:
        try:
            datos_ecg = pd.read_csv(file, sep=delim)
            if datos_ecg.shape[1] >= 2:  # Verifica que haya al menos dos columnas
                return datos_ecg
        except pd.errors.EmptyDataError:
            print('El archivo está vacío.')
            return None
        except pd.errors.ParserError:
            print('Error de análisis al leer el archivo.')
            return None
        except Exception as e:
            print('No se puede leer el archivo: ', str(e))
    print(f'Formato de archivo no soportado: {file.name}')
    return None
def filtro_pasa_bajas(v, fs):
    coeficientes =  firwin(numtaps = 101, cutoff = 40, fs = fs)
    V_FIL =  lfilter(coeficientes, 1.0, v)
 #   umbral = np.max(señal_FPB)*0.1 #porcentaje de umbral a partir del punto R másicmo 
    return (V_FIL)

def algoritmo_segunda_derivada(v):#tiene entrada la señal filtrada
    coeficientes = np.array([1,1,1,1,-10,1,1,1,1])
    ECG_A = 4 * (np.convolve(v, coeficientes, mode= 'same' ))
    umbral = np.max(ECG_A)
    return (ECG_A*ECG_A), umbral

    #ECG_A = (4 * (v[i+4] + v[i+3] + v[i+2] + v[i+1] - 10 * v[i] + v[i-1] + v[i-2] + v[i-3] + v[i-4]))
def detección_RR(V, d, U, T, FC):#Tiene como entrada la señal aplicada la segunda derivada 
    output = ecg.ecg(signal=V, sampling_rate=FC, show=False)
    picosRR = output['rpeaks']
    #picosRR = find_peaks(V, distance = d, height = U)
    #picosRR = picosRR[0]
    intervalosRR = np.diff(T[picosRR])
    return picosRR, intervalosRR*1000

def Parametro_dominio_tiempo(intervalos_RR):
    resultadosDT = time_domain.time_domain(intervalos_RR)
    return resultadosDT
def Parametro_dominio_frecuencia(RR, intervalos_RR, T):
    #Interpolar el Tacograma para Obtener un Muestreo Regular
    RR = np.array(RR)
    intervalos_RR = np.array(intervalos_RR)
    T = np.array(T)
    fs = 4
    tiempo_regular = np.arange(T[RR][0],T[RR][-1], 1 /fs ) 
    interp_func = interp1d(T[RR][1:],intervalos_RR, kind='linear', fill_value='extrapolate')
    rr_uniforme = interp_func(tiempo_regular)
    #Eliminar tendencias 
    tendencia = np.polyfit(tiempo_regular, rr_uniforme, 1)
    rr_det = rr_uniforme - np.polyval(tendencia, tiempo_regular)
    return rr_det, fs

def FFT(RR, FS):
    nfft = len(RR)  # Número de puntos de la FFT       #2949
    rr_fft = np.fft.rfft(RR)
    freqs_RR = np.fft.rfftfreq(nfft, 1 / FS)
    rr_fft = np.abs(rr_fft) # Magnitud de la FFT
    return freqs_RR, rr_fft


def calcular_potencia_banda(frecuencias, psd, banda):
    indices = np.where((frecuencias >= banda[0]) & (frecuencias <= banda[1]))[0]
    potencia = np.trapz(psd[indices], frecuencias[indices])
    frecuencia_pico = frecuencias[indices][np.argmax(psd[indices])] if indices.size > 0 else 0
    return potencia, frecuencia_pico

# Cálculo de parámetros de dominio de frecuencia usando Welch
def calculo_welch(RR, FS):
    # Estimación PSD con el método de Welch
    frecuencias_welch, psd = welch(RR, fs=FS, window='hamming', nperseg=1024, noverlap=512, nfft=2048)
    
    # Definición de bandas de frecuencia
    banda_vlf = (0.0033, 0.04)
    banda_lf = (0.04, 0.15)
    banda_hf = (0.15, 0.4)
    
    # Calcular potencia y frecuencia pico en cada banda
    potencia_vlf, pico_vlf = calcular_potencia_banda(frecuencias_welch, psd, banda_vlf)
    potencia_lf, pico_lf = calcular_potencia_banda(frecuencias_welch, psd, banda_lf)
    potencia_hf, pico_hf = calcular_potencia_banda(frecuencias_welch, psd, banda_hf)
    
    # Potencia total
    potencia_total = potencia_vlf + potencia_lf + potencia_hf
    
    # Calcular el porcentaje de cada banda
    porcentaje_vlf = (potencia_vlf / potencia_total) * 100
    porcentaje_lf = (potencia_lf / potencia_total) * 100
    porcentaje_hf = (potencia_hf / potencia_total) * 100
    
    # Potencia en unidades normalizadas
    potencia_lf_nu = (potencia_lf / (potencia_lf + potencia_hf)) * 100
    potencia_hf_nu = (potencia_hf / (potencia_lf + potencia_hf)) * 100
    
    # Cociente LF/HF
    lf_hf_ratio = potencia_lf / potencia_hf if potencia_hf != 0 else 0
    
    # Potencia en escala logarítmica
    potencia_vlf_log = np.log(potencia_vlf) if potencia_vlf > 0 else 0
    potencia_lf_log = np.log(potencia_lf) if potencia_lf > 0 else 0
    potencia_hf_log = np.log(potencia_hf) if potencia_hf > 0 else 0

    # Imprimir resultados
    print(f'VLF - Potencia: {potencia_vlf:.4f} ms², Frecuencia Pico: {pico_vlf:.4f} Hz, Potencia (log): {potencia_vlf_log:.4f}, Porcentaje: {porcentaje_vlf:.2f}%')
    print(f'LF - Potencia: {potencia_lf:.4f} ms², Frecuencia Pico: {pico_lf:.4f} Hz, Potencia (log): {potencia_lf_log:.4f}, Porcentaje: {porcentaje_lf:.2f}%')
    print(f'HF - Potencia: {potencia_hf:.4f} ms², Frecuencia Pico: {pico_hf:.4f} Hz, Potencia (log): {potencia_hf_log:.4f}, Porcentaje: {porcentaje_hf:.2f}%')
    print(f'Potencia total: {potencia_total:.4f} ms²')
    print(f'Potencia LF (nu): {potencia_lf_nu:.2f}')
    print(f'Potencia HF (nu): {potencia_hf_nu:.2f}')
    print(f'Ratio LF/HF: {lf_hf_ratio:.4f}')

    # Retornar los valores calculados si deseas usarlos en otros lugares
    return {
        "potencia_vlf": potencia_vlf,
        "pico_vlf": pico_vlf,
        "potencia_vlf_log": potencia_vlf_log,
        "porcentaje_vlf": porcentaje_vlf,
        "potencia_lf": potencia_lf,
        "pico_lf": pico_lf,
        "potencia_lf_log": potencia_lf_log,
        "porcentaje_lf": porcentaje_lf,
        "potencia_hf": potencia_hf,
        "pico_hf": pico_hf,
        "potencia_hf_log": potencia_hf_log,
        "porcentaje_hf": porcentaje_hf,
        "potencia_total": potencia_total,
        "potencia_lf_nu": potencia_lf_nu,
        "potencia_hf_nu": potencia_hf_nu,
        "lf_hf_ratio": lf_hf_ratio
    }, frecuencias_welch, psd














# Vista para mostrar la página de inicio
def home(request):
    return render(request, 'home.html')

# Vista para cerrar la sesión de un usuario
@login_required
def signout(request):
    logout(request)
    return redirect('home')

# Vista para el inicio de sesión
def signin(request):
    
    if request.method == 'GET':
        return render(request, 'signin.html', {"form": AuthenticationForm()})
    else:
        user = authenticate(
            request, username=request.POST['username'], password=request.POST['password1'])
        if user is None:
            return render(request, 'signin.html', {"form": AuthenticationForm(), "error": "Nombre de usuario o contraseña incorrectos."})

        login(request, user)
        return redirect('pacientes')

# Vista para mostrar los detalles de un paciente específico


# Vista para marcar un paciente como completado
@login_required
def complete_paciente(request, paciente_id):
    paciente = get_object_or_404(Paciente, pk=paciente_id, user=request.user)
    if request.method == 'POST':
        paciente.fecha_nacimiento = timezone.now()
        paciente.save()
        return redirect('pacientes')


# Vista del perfil del especialista
@login_required
def perfil_doc(request):
    user = request.user
    try:
        doctor = Especialista.objects.get(user=user)
        context = {
            'nombre_especialista': doctor.nombre_especialista,
            'apellido_paterno': doctor.apellido_paterno,
            'apellido_materno': doctor.apellido_materno,
            'fecha_nacimiento': doctor.fecha_nacimiento,
            'departamento': doctor.departamento_id,  # Cambié a la forma correcta
            'username': user.username,
        }
        return render(request, 'perfil_especialista.html', context)
    except Especialista.DoesNotExist:
        return redirect('error_page')

def error_page(request):
    return render(request, 'error_page.html')

@login_required
def homeDoctor(request):
    especialista = Especialista.objects.get(user=request.user)  # Obtiene el especialista actual
    return render(request, 'homeDoctor.html', {'especialista': especialista})





@login_required

def ver_grafico(request, paciente_id):
    paciente = get_object_or_404(Paciente, id_paciente=paciente_id)  # Asegúrate de que 'id_paciente' sea el nombre correcto del campo
    
    tiempo_mostrar = int(request.GET.get('tiempo', 20))
    banda = int(request.GET.get('banda', 10000))  # Valor predeterminado de 10000 muestras
    amplitu = int(request.GET.get('amplitud', 3))
    banda_tacograma = int(request.GET.get('banda_tacograma', 100))  # Valor predeterminado para tacograma

    try:
        
        registro_ecg = get_object_or_404(ECG, id_ecg=paciente_id)
    except Exception as e:
        print(f"Error al obtener el registro ECG: {e}")
        return render(request, 'ver_grafico.html', {'mensaje': 'No se pudo encontrar el registro ECG.'})

    ecg_path = registro_ecg.archivo_ecg.path
    datos_ecg = formato_ecg(ecg_path)

    if datos_ecg is None:
        return render(request, 'ver_grafico.html', {'mensaje': 'El archivo ECG no pudo ser leído.'})

    if datos_ecg.shape[1] < 2:
        return render(request, 'ver_grafico.html', {'mensaje': 'El archivo ECG no tiene la estructura correcta.'})


    datos_ecg.iloc[:, 0] = pd.to_numeric(datos_ecg.iloc[:, 0], errors='coerce')
    datos_ecg.iloc[:, 1] = pd.to_numeric(datos_ecg.iloc[:, 1], errors='coerce')

    tiempo_ECG = datos_ecg.iloc[:, 0]
    voltaje_ECG = datos_ecg.iloc[:, 1]
    fm = 1 /(tiempo_ECG.iloc[1]-tiempo_ECG.iloc[0])
    distancia = fm * 0.8
    voltaje = filtro_pasa_bajas(voltaje_ECG, fm)
    voltaje, umbral = algoritmo_segunda_derivada(voltaje)
    picosRR, intervalosRR = detección_RR(voltaje, distancia, umbral, tiempo_ECG, fm)
    HR = (60/intervalosRR)*1000 # Frecuencia cardiaca
    print(f'FRECUECNIA CARDICA: {HR}')

    resultadosDT = Parametro_dominio_tiempo(intervalosRR)
    #print(resultadosDT)  # Muestra el diccionario para verificar valores

    RRN, fs = Parametro_dominio_frecuencia(picosRR, intervalosRR, tiempo_ECG)
    freqs_RR, rr_fft = FFT(RRN, fs)
    resultadosDF, frecuencias_welch, psd = calculo_welch(RRN, fs)


 # Direccionamiento de los parámetros a los modelos 
    analisis_tiempo, created = AnalisisDominioTiempo.objects.get_or_create(
    ecg=registro_ecg,
    defaults={
        'nni_mean': round(resultadosDT['nni_mean'], 2),
        'nni_min': round(resultadosDT['nni_min'], 2),
        'nni_max': round(resultadosDT['nni_max'], 2),
        'hr_mean': round(resultadosDT['hr_mean'], 2),
        'hr_min': round(resultadosDT['hr_min'], 2),
        'hr_max': round(resultadosDT['hr_max'], 2),
        'std_hr': round(resultadosDT['hr_std'], 2),
        'nni_diff_mean': round(resultadosDT['nni_diff_mean'], 2),
        'nni_diff_min': round(resultadosDT['nni_diff_min'], 2),
        'nni_diff_max': round(resultadosDT['nni_diff_max'], 2),
        'sdnn': round(resultadosDT['sdnn'], 2),
        'sdnn_index': round(resultadosDT['sdnn_index'], 2),
        'sdann': round(resultadosDT['sdann'], 2),
        'rmssd': round(resultadosDT['rmssd'], 2),
        'sdsd': round(resultadosDT['sdsd'], 2),
        'nn50': round(resultadosDT['nn50'], 2),
        'pnn50': round(resultadosDT['pnn50'], 2),
        'nn20': round(resultadosDT['nn20'], 2),
        'rr_mean': round(float(np.mean(intervalosRR)), 2),
        'total_intervalos_rr': len(intervalosRR)
    }
)


# Dirección de los parámetros a los modelos 
    analisis_frecuencia = AnalisisDominioFrecuencia.objects.get_or_create(
        ecg=registro_ecg,
        potencia_vlf=resultadosDF['potencia_vlf'],
        pico_vlf=resultadosDF['pico_vlf'],
        potencia_vlf_log=resultadosDF['potencia_vlf_log'],
        porcentaje_vlf=resultadosDF['porcentaje_vlf'],
        potencia_lf=resultadosDF['potencia_lf'],
        pico_lf=resultadosDF['pico_lf'],
        potencia_lf_log=resultadosDF['potencia_lf_log'],
        porcentaje_lf=resultadosDF['porcentaje_lf'],
        potencia_hf=resultadosDF['potencia_hf'],
        pico_hf=resultadosDF['pico_hf'],
        potencia_hf_log=resultadosDF['potencia_hf_log'],
        porcentaje_hf=resultadosDF['porcentaje_hf'],
        potencia_total=resultadosDF['potencia_total'],
        potencia_lf_nu=resultadosDF['potencia_lf_nu'],
        potencia_hf_nu=resultadosDF['potencia_hf_nu'],
        lf_hf_ratio=resultadosDF['lf_hf_ratio']
    )

    # JSON para el frontend
    datos_completos = {
        "ecg": {
            "tiempo": tiempo_ECG.tolist(),
            "voltaje": voltaje_ECG.tolist(),
        },
        "tacograma": {
            "picos": picosRR.tolist(),
            "intervalos": intervalosRR.tolist(),
        }
    }

    # Datos iniciales para gráficos
    tiempo_mostrar = tiempo_ECG[:banda]
    voltaje_mostrar = voltaje_ECG[:banda]

    fig_ecg = {
        'data': [
            {'x': tiempo_mostrar.tolist(), 'y': voltaje_mostrar.tolist(), 'type': 'scatter', 'mode': 'lines', 'name': 'ECG'}
        ],
        'layout': {
            'title': 'Gráfico ECG',
            'xaxis': {'title': 'Tiempo'},
            'yaxis': {'title': 'Voltaje'}
        }
    }

    fig_tacograma = {
        'data': [
            {'x': picosRR.tolist(), 'y': intervalosRR.tolist(), 'type': 'scatter', 'mode': 'lines+markers', 'name': 'Tacograma'}
        ],
        'layout': {
            'title': 'Tacograma - Intervalos RR',
            'xaxis': {'title': 'Tiempo (s)'},
            'yaxis': {'title': 'Intervalo RR (ms)'}
        }
    }
    fig_histogramaRR = {
        'data': [
            {'x': intervalosRR.tolist(), 'type': 'histogram'}
        ],
        'layout': {
            'title': 'Histograma - Intervalos RR',
            'xaxis': {'title': 'Intervalo RR (ms)'},
            'yaxis': {'title': 'Frecuencia'}
        }
    }
    
    fig_histogramaHR = {
        'data' : [
            {'x': HR.tolist(), 'type': 'histogram'}
        ],
        'layout': {
            'title': 'Histograma Frecuencia Cardiaca',
            'xaxis':{'title': 'FC (lpm)'},
            'yaxis': {'title': 'Frecuencia'}
        },
    }

    fig_FFT = {
        'data': [
            {'x': freqs_RR.tolist(), 'y': rr_fft.tolist(), 'type': 'scatter', 'mode': 'lines+markers', 'name': 'FFT ECG'}
        ],
        'layout': {
            'title': 'FFT ',
            'xaxis': {'title': 'Frecuencia (Hz)'},
            'yaxis': {'title': 'Amplitud (ms)'}
        }
    }

    fig_welch = {
        'data': [
            {'x': frecuencias_welch.tolist(), 'y': psd.tolist(), 'type': 'scatter', 'mode': 'lines+markers', 'name': 'Welch ECG'}
        ],
        'layout': {
            'title': 'Welch ',
            'xaxis': {'title': 'Frecuencia (Hz)'},
            'yaxis': {'title': 'Amplitud (ms2/Hz)'}
        }
    }


    length_ecg = len(fig_ecg['data'][0]['x'])

    return render(request, 'ver_grafico.html', {
        'registro_ecg': registro_ecg,
        'graph_json': json.dumps(fig_ecg),
        'graph_json_tacograma': json.dumps(fig_tacograma),
        'datos_completos_json': json.dumps(datos_completos),
        'banda': banda,
        'length_ecg': length_ecg,
        'analisis_tiempo': analisis_tiempo,
        'graph_json_RR': json.dumps(fig_histogramaRR),
        'graph_json_HR': json.dumps(fig_histogramaHR),
        'graph_json_FFT': json.dumps(fig_FFT),
        'graph_json_Welch': json.dumps(fig_welch)
        
    })
    
    
def generar_pdf(request):
    # Renderizar el contenido HTML del template
    html_content = render_to_string('ver_grafico.html', {'is_pdf': True})

    # Crear la respuesta HTTP para el PDF
    response = HttpResponse(content_type='application/pdf')
    response['Content-Disposition'] = 'attachment; filename="informe.pdf"'

    # Generar el PDF desde el contenido HTML
    HTML(string=html_content).write_pdf(response)

    return response

from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.models import User
from django.db import IntegrityError
from django.utils import timezone
from django.contrib.auth.decorators import login_required
from .models import *
from .forms import *


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
    pacientes = Paciente.objects.filter(especialista=request.user.especialista)
    return render(request, 'paciente.html', {"pacientes": pacientes})



def editar_paciente(request, id):
    paciente = get_object_or_404(Paciente, id_paciente=id)
    if request.method == 'POST':
        Formulario = PacienteForm(request.POST, request.FILES, instance=paciente)  # request.FILES para manejar archivos
        if Formulario.is_valid():
            Formulario.save()
            return redirect('pacientes')  # Cambia a la vista que necesites después de guardar
    else:
        Formulario = PacienteForm(instance=paciente)
    return render(request, 'pacientes/editar_paciente.html', {"form": PacienteForm()})

'''En la plantilla editar.html, puedes usar el formulario de la siguiente manera:
#{'Formulario': Formulario} significa que dentro de la plantillaeditar.html, puedes acceder a la instancia del formulario con la variable Formulario.'''



def eliminar_paciente(request, id):
    pacientes = get_object_or_404(Paciente, id_paciente=id) # referencia al campo de la clase 
    pacientes.delete() # Elimina el paciente
    return redirect('pacientes') # Redirige a la lista de pacientes





# Vista para mostrar los pacientes completados
@login_required
def pacientes_completed(request):
    pacientes = Paciente.objects.filter(user=request.user, fecha_nacimiento__isnull=False).order_by('-fecha_nacimiento')
    return render(request, 'paciente_completed.html', {"pacientes": pacientes})  # Usa una plantilla diferente

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
@login_required
def paciente_detail(request, paciente_id):
    paciente = get_object_or_404(Paciente, pk=paciente_id, user=request.user)
    if request.method == 'GET':
        form = PacienteForm(instance=paciente)
        return render(request, 'paciente_detail.html', {'paciente': paciente, 'form': form})
    else:
        form = PacienteForm(request.POST, instance=paciente)
        if form.is_valid():
            form.save()
            return redirect('pacientes')
        return render(request, 'paciente_detail.html', {'paciente': paciente, 'form': form, 'error': 'Error al actualizar al paciente.'})

# Vista para marcar un paciente como completado
@login_required
def complete_paciente(request, paciente_id):
    paciente = get_object_or_404(Paciente, pk=paciente_id, user=request.user)
    if request.method == 'POST':
        paciente.fecha_nacimiento = timezone.now()
        paciente.save()
        return redirect('pacientes')

# Vista para eliminar un paciente
@login_required
def delete_paciente(request, paciente_id):
    paciente = get_object_or_404(Paciente, pk=paciente_id, user=request.user)
    if request.method == 'POST':
        paciente.delete()
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

def homeDoctor(request):
    return render(request, 'homeDoctor.html')

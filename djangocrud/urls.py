"""djangocrud URL Configuration

Este archivo configura las URL de tu proyecto Django. Las URLs se asignan a vistas específicas que manejan las solicitudes HTTP.

Para más información sobre cómo funcionan las configuraciones de URL en Django, consulta la documentación oficial:
https://docs.djangoproject.com/en/4.1/topics/http/urls/

Ejemplos:
- Vistas de función:
    1. Agrega una importación: from my_app import views
    2. Agrega una URL a urlpatterns: path('', views.home, name='home')
    
- Vistas basadas en clase:
    1. Agrega una importación: from other_app.views import Home
    2. Agrega una URL a urlpatterns: path('', Home.as_view(), name='home')
    
- Incluyendo otro archivo de configuración de URL:
    1. Importa la función include: from django.urls import include, path
    2. Agrega una URL a urlpatterns: path('blog/', include('blog.urls'))
"""

# Importa el módulo admin de django.contrib, que permite administrar el sitio.
from django.contrib import admin

# Importa las funciones path de django.urls, que se utilizan para definir las rutas de URL.
from django.urls import path

# Importa las vistas desde el módulo pacientes.
from paciente import views
# La lista urlpatterns contiene las rutas de URL y las vistas asociadas.
urlpatterns = [
    # Ruta para la página de inicio del sitio, asignada a la vista 'home'.
    path('', views.home, name='home'),
    
    
    
    path('homeDoctor/', views.homeDoctor, name='homeDoctor'),    
    # Ruta para el panel de administración de Django.
    path('admin/', admin.site.urls),
    
    # Ruta para el registro de nuevos usuarios, asignada a la vista 'signup'.
    path('signup/', views.signup, name='signup'),
    
    # Ruta para visualizar todos los pacientes, asignada a la vista 'pacientes'.
    path('pacientes/', views.pacientes, name='pacientes'),
    
    # Ruta para visualizar los registros de pacientes completados, asignada a la vista 'pacientes_completed'.
    path('pacientes_completed/', views.pacientes_completed, name='pacientes_completed'),
    
    # Ruta para cerrar sesión, asignada a la vista 'signout'.
    path('logout/', views.signout, name='logout'),
    
    # Ruta para iniciar sesión, asignada a la vista 'signin'.
    path('signin/', views.signin, name='signin'),
    
    # Ruta para crear un nuevo registro de paciente, asignada a la vista 'create_paciente'.
    path('create_paciente/', views.create_paciente, name='create_paciente'),
    
    # Ruta para ver los detalles de un paciente específico, utilizando el ID del paciente.
    path('pacientes/<int:paciente_id>', views.paciente_detail, name='paciente_detail'),
    
    # Ruta para marcar un paciente como completado, utilizando el ID del paciente.
    path('paciente/<int:paciente_id>/complete', views.complete_paciente, name='complete_paciente'),
    
    # Ruta para eliminar un paciente específico, utilizando el ID del paciente.
    path('pacientes/<int:paciente_id>/eliminar', views.eliminar_paciente, name='eliminar_paciente'),
    
    
    
    # Ruta para editar un paciente específico, utilizando el ID del paciente.
    path('pacientes/<int:paciente_id>/editar', views.editar_paciente, name='editar_paciente'),

    
    path('perfil_especialista/', views.perfil_doc, name='perfilEspecialista'), # Nueva vista para datos personales  
    path('error/', views.error_page, name='error_page'),  # Agrega esta línea
    ]   
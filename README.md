# Simulador de Clima Polymarket (3 Bots en Paralelo)

Este proyecto ejecuta una simulación en tiempo real (Paper Trading) de tres bots basados en modelos predictivos XGBoost que compiten en los mercados de temperatura diaria de Madrid y Londres de Polymarket.

Toda la infraestructura funciona de manera **100% gratuita y en la nube**, sin necesidad de tener tu ordenador encendido a las 06:00 AM.

---

## 🛠️ Cómo Subir y Activar el Proyecto en GitHub (Paso a Paso)

Sigue estos sencillos pasos para publicar tu simulador y ver el dashboard web desde cualquier dispositivo:

### Paso 1: Crear un Repositorio en GitHub
1. Entra a [github.com](https://github.com/) e inicia sesión.
2. Haz clic en **New repository** (o ve a [github.com/new](https://github.com/new)).
3. Dale un nombre (por ejemplo: `polymarket-weather-simulator`).
4. Puedes elegir configurarlo como **Private** (Privado) o **Public** (Público). Ambos funcionan perfectamente y de forma gratuita con GitHub Actions.
5. Deja las opciones de inicialización vacías (sin README, gitignore ni licencia) y haz clic en **Create repository**.

### Paso 2: Subir el Código desde tu Ordenador
Abre tu terminal en el directorio `prueba_real_interfaz_y_3_modelos` y ejecuta:

```bash
# Inicializar Git en la carpeta
git init

# Añadir todos los archivos
git add .

# Hacer el primer commit
git commit -m "Estructura inicial del simulador"

# Renombrar la rama principal a main
git branch -M main

# Vincular tu repositorio local con GitHub (reemplaza por la URL de tu repo)
git remote add origin https://github.com/TU_USUARIO/TU_REPOSITORIO.git

# Subir los archivos
git push -u origin main
```

### Paso 3: Dar Permisos de Escritura a GitHub Actions (¡CRÍTICO!)
Por defecto, GitHub Actions tiene permisos de solo lectura y el bot fallará al intentar guardar los balances de los bots. Debes activar los permisos de escritura:
1. En tu repositorio de GitHub, ve a la pestaña **Settings** (Configuración) en la barra superior.
2. En el menú lateral izquierdo, ve a **Actions** > **General**.
3. Baja hasta la sección **Workflow permissions**.
4. Selecciona la opción **Read and write permissions** (Permisos de lectura y escritura).
5. Marca la casilla **Allow GitHub Actions to create and approve pull requests** si estuviera disponible.
6. Haz clic en **Save** (Guardar).

### Paso 4: Activar la Web del Dashboard (GitHub Pages)
1. En la pestaña **Settings** de tu repositorio, ve a **Pages** (en el menú lateral izquierdo).
2. En la sección **Build and deployment** > **Branch**:
   * Cambia la opción de `None` a `main`.
   * Cambia la carpeta de `/ (root)` a `/docs` (donde está alojado el Dashboard).
3. Haz clic en **Save** (Guardar).
4. ¡Listo! En unos 2 minutos, GitHub te dará un enlace público (tipo `https://tu_usuario.github.io/tu_repositorio/`) donde podrás entrar a ver tu dashboard interactivo.

---

## 🚀 ¿Cómo se Ejecuta el Bot?

* **Ejecución Automática:** El archivo de configuración `.github/workflows/simulation.yml` despertará al bot en la nube automáticamente todas las mañanas a las **06:00 AM** (hora de Madrid).
* **Ejecución Manual (Para probar ahora mismo):**
  1. Ve a la pestaña **Actions** en tu repositorio de GitHub.
  2. Haz clic en el flujo **Polymarket Weather Simulation** en el panel izquierdo.
  3. Haz clic en el botón desplegable **Run workflow** a la derecha y pulsa el botón verde **Run workflow**. 
  4. Esto entrenará los 3 modelos e intentará buscar los mercados de Polymarket de hoy y simular las compras en unos 2 minutos.

---

## 📈 ¿Qué Modelos están compitiendo?
1. **Bot V1 (Base):** Entrenado con 3.0 años de historial real, profundidad de árbol de 4, sin variables de tendencia.
2. **Bot A (Máximo ROI):** Entrenado con 2.5 años de historial real, profundidad de árbol de 4, sin variables de tendencia.
3. **Bot B (Máximo Acierto):** Entrenado con 2.0 años de historial real, profundidad de árbol de 3, utilizando variables avanzadas de tendencia.

# Proyecto Final - Métodos de Optimización

Aplicación web en Streamlit para encontrar el mínimo de una función usando:

- Método del gradiente
- Método del gradiente conjugado
- Método de Newton
- Búsqueda de línea con primera y segunda condición de Wolfe

## Ejecución local

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Despliegue recomendado

1. Subir `app.py`, `requirements.txt` y `README.md` a un repositorio de GitHub.
2. Entrar a Streamlit Community Cloud.
3. Crear una nueva app usando el repositorio.
4. Seleccionar `app.py` como archivo principal.
5. Compartir el enlace público con el profesor.

## Ejemplos de prueba

Función: `x1**2 + x2**2`  
Punto inicial: `3, 4`

Función: `(x1-1)**2 + (x2+2)**2`  
Punto inicial: `0, 0`

Función: `100*(x2-x1**2)**2 + (1-x1)**2`  
Punto inicial: `-1.2, 1`

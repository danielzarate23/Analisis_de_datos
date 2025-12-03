# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.11.9
FROM python:${PYTHON_VERSION}-slim as base

# No generar archivos .pyc
ENV PYTHONDONTWRITEBYTECODE=1
# Logueo sin buffer
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Usuario no privilegiado (buena práctica, opcional)
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser

# Copiar e instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código de la app
COPY . .

# Usar el usuario no privilegiado
USER appuser

# Puerto por defecto de Streamlit
EXPOSE 8501

# Ejecutar Streamlit
CMD ["streamlit", "run", "app_streamlit_vendedores.py", "--server.port=8501", "--server.address=0.0.0.0"]

# POLUX Backend

Este es el backend de POLUX con FastAPI + LangChain + Chroma, conectado a GPT-4 Turbo.

## Endpoints

- POST `/ask`: recibe una pregunta y devuelve una respuesta basada en la base vectorial.

## Configuración

1. Crea un archivo `.env` basado en `.env.example`
2. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```
3. Ejecuta el servidor localmente:
   ```bash
   uvicorn main:app --reload
   ```

4. Crea y llena tu base vectorial en `./chroma_db/`
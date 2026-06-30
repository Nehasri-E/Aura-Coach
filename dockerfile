# -------- FRONTEND BUILD (React) --------
FROM node:18 AS frontend-build

WORKDIR /app

COPY package*.json ./
RUN npm install

COPY . .
RUN npm run build


# -------- BACKEND (FastAPI) --------
FROM python:3.10-slim

WORKDIR /app

COPY backend /app/backend
COPY data /app/data
COPY --from=frontend-build /app/build /app/backend/build

COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# 🧠 Train both ML models at build time so the image ships with
# backend/models/*.joblib already present. If data/raw has fewer than
# 10 sessions, the performance model step will fail the build on
# purpose — train it locally first and commit the .joblib files, or
# seed data/raw with enough sample sessions before building.
RUN python backend/ml/train_performance_model.py && \
    python backend/ml/train_tone_model.py

EXPOSE 8080

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8080"]

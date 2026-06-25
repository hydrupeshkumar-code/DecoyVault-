FROM node:20-alpine AS builder

WORKDIR /app
COPY frontend/package.json ./
RUN npm install

COPY frontend/ ./
ARG REACT_APP_API_URL=http://localhost:8000
ENV REACT_APP_API_URL=$REACT_APP_API_URL
RUN npm run build

# Production stage – serve via nginx
FROM nginx:1.25-alpine
# Vite emits the production bundle to /app/dist (not /app/build, which is CRA's
# default). Copying the wrong directory makes `docker build` fail with
# "COPY failed: ... /app/build: not found".
COPY --from=builder /app/dist /usr/share/nginx/html
COPY docker/nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]

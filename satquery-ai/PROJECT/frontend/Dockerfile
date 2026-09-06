# SatQuery AI - Frontend Dockerfile
# Base: Production Alpine Nginx
FROM nginx:alpine

LABEL maintainer="SatQuery AI Team <team@satquery.ai>"
LABEL description="SatQuery AI Mission Control Dashboard Nginx Server"

# Remove default nginx static assets
RUN rm -rf /usr/share/nginx/html/*

# Copy static frontend SPA and assets
COPY index.html /usr/share/nginx/html/index.html

# Expose standard HTTP port
EXPOSE 80

# Start Nginx server in foreground
CMD ["nginx", "-g", "daemon off;"]

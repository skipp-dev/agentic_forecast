#!/bin/bash

# Generate self-signed SSL certificates for development
# Run this script from the deployment/ssl directory

set -e

CERT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$CERT_DIR"

echo "Generating self-signed SSL certificates..."

# Generate private key
openssl genrsa -out key.pem 2048

# Generate certificate signing request
openssl req -new -key key.pem -out cert.csr -subj "/C=US/ST=State/L=City/O=IB Forecast/CN=localhost"

# Generate self-signed certificate
openssl x509 -req -days 365 -in cert.csr -signkey key.pem -out cert.pem

# Clean up CSR file
rm cert.csr

# Set appropriate permissions
chmod 600 key.pem
chmod 644 cert.pem

echo "SSL certificates generated successfully!"
echo "Certificate: $CERT_DIR/cert.pem"
echo "Private key: $CERT_DIR/key.pem"
echo ""
echo "Note: These are self-signed certificates for development only."
echo "For production, use certificates from a trusted CA."
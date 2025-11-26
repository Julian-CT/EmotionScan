# This file helps Nixpacks detect Python
# The actual application is in api/api_server.py
import sys
import os

# Redirect to the actual app
if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'api'))
    from api_server import app
    import os
    
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)


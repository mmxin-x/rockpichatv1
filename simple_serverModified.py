#!/usr/bin/env python3
"""
Simple RKLLM Server - OpenAI API Compatible

This server provides an OpenAI-compatible API for the RKLLM runtime.
It's a simplified version that preserves the working implementation
from the original server.py while fitting into the modular structure.

This simplified bridge implementation provides:
1. OpenAI-compatible API endpoints (/v1/models, /v1/chat/completions)
2. Full tool calling support
3. Streaming and non-streaming modes
4. Environment variable configuration

While development continues on the fully modularized version,
this server provides a stable and working solution.
"""

import ctypes
import os
import sys
import time
import json
import uuid
import threading
import queue
import re
import traceback
from flask import Flask, request, Response, jsonify, stream_with_context
from flask_cors import CORS
from flask import current_app

# Import modules from the new structure
try:
    from rkllm.config import SERVER_HOST, SERVER_PORT, API_BASE_PATH, DEBUG_MODE, MODEL_PATH, LIB_PATH
except ImportError:
    print("Warning: Could not import from rkllm.config. Using fallback from server.py.")

# Import original server code
try:
    from server import (
        init_rkllm_model, chat_completions_handler, 
        list_models, format_openai_error_response
    )
except ImportError:
    print("Error: Could not import required functions from server.py")
    sys.exit(1)

# Create Flask app
app = Flask(__name__)
CORS(app)

# Add error handler for all exceptions
@app.errorhandler(Exception)
def handle_exception(e):
    print(f"Error occurred: {str(e)}")
    traceback.print_exc()
    return format_openai_error_response(str(e), 500)

# API routes
@app.route('/v1/models', methods=['GET'])
def models_route():
    try:
        return list_models()
    except Exception as e:
        return format_openai_error_response(f"Error listing models: {str(e)}", 500)

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions_route():
    try:
        return chat_completions_handler()
    except Exception as e:
        return format_openai_error_response(f"Error in chat completions: {str(e)}", 500)

#New endpoint to reset the loaded RKLLM model
@app.post("/v1/chat/reset")
def reset_context():
	try:
		success = init_rkllm_model()
		if not success:
			return format_openai_error_response("failed to reset RKLLM model", 500)
		return jsonify({"status":"reset"}),200
	except Exception as e:
		return format_openai_error_response(str(e), 500)

# Main entry point
if __name__ == '__main__':
    # Try to use configuration from different sources with fallbacks
    try:
        # Try imports from both locations
        server_config_imported = False
        rkllm_config_imported = False
        
        try:
            # First try the modular config
            from rkllm.config import SERVER_HOST, SERVER_PORT, API_BASE_PATH, DEBUG_MODE
            rkllm_config_imported = True
            print("Using configuration from rkllm.config")
        except ImportError:
            # Fall back to original config
            try:
                from config import SERVER_HOST, SERVER_PORT, API_BASE_PATH, DEBUG_MODE
                server_config_imported = True
                print("Using configuration from config.py")
            except ImportError:
                # Final fallback to server.py config
                from server import SERVER_HOST, SERVER_PORT, API_BASE_PATH, DEBUG_MODE
                print("Using configuration from server.py")
        
        # Initialize the model and start the server
        if init_rkllm_model():
            print(f"Starting Flask server for RKLLM OpenAI-compliant API on http://{SERVER_HOST}:{SERVER_PORT}{API_BASE_PATH}/chat/completions")
            app.run(host=SERVER_HOST, port=SERVER_PORT, threaded=True, debug=DEBUG_MODE)
        else:
            print("Failed to initialize RKLLM model. Server not starting.")
            sys.exit(1)
    
    except Exception as e:
        print(f"Error starting server: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

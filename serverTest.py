""" IMPORTS AND CONFIGS """
import ctypes
import os
import time
import json
import uuid
import threading
import queue 
import re 
from flask import Flask, request, Response, jsonify, stream_with_context
from config import *


# LLMCallState (Page 2)
LLM_RUN_NORMAL = 0
LLM_RUN_FINISH = 1
LLM_RUN_WAITING = 2 # Not explicitly numbered, assuming sequence
LLM_RUN_ERROR = 3   # Not explicitly numbered, assuming sequence

# RKLLMInputType (Page 8)
RKLLM_INPUT_PROMPT = 0
RKLLM_INPUT_TOKEN = 1
RKLLM_INPUT_EMBED = 2
RKLLM_INPUT_MULTIMODAL = 3

# RKLLMInferMode (Page 10)
RKLLM_INFER_GENERATE = 0
RKLLM_INFER_GET_LOGITS = 1

# --- Global State & Configuration ---
rkllm_lib = None
llm_handle = ctypes.c_void_p()
rkllm_params_global = None
model_initialized = False

default_history = ""
conversation_history = default_history
conversation_lock = threading.Lock()

thread_queue_map = {}
thread_queue_map_lock = threading.Lock() # Lock for critical RKLLM sections and shared state
# stream_token_queue = queue.Queue()
active_streams = {}
active_streams_lock = threading.Lock()


app = Flask(__name__)


""" DEFINE DATA STRUCTURES """

class RKLLMExtendParam(ctypes.Structure):
    """ Corresponds to RKLLMExtendParam structure """
    _fields_ = [
        ("base_domain_id", ctypes.c_int32),
        ("embed_flash", ctypes.c_int8),
        ("enabled_cpus_num", ctypes.c_int8),
        ("enabled_cpus_mask", ctypes.c_uint32),
        ("n_batch", ctypes.c_uint8),
        ("use_cross_attn", ctypes.c_int8),
        ("reserved", ctypes.c_uint8 * 104)
    ]

class RKLLMParam(ctypes.Structure):
    """ Corresponds to RKLLMParam structure  """
    _fields_ = [
        ("model_path", ctypes.c_char_p),
        ("max_context_len", ctypes.c_int32),
        ("max_new_tokens", ctypes.c_int32),
        ("top_k", ctypes.c_int32),
        ("n_keep", ctypes.c_int32),
        ("top_p", ctypes.c_float),
        ("temperature", ctypes.c_float),
        ("repeat_penalty", ctypes.c_float),
        ("frequency_penalty", ctypes.c_float),
        ("presence_penalty", ctypes.c_float),
        ("mirostat", ctypes.c_int32),
        ("mirostat_tau", ctypes.c_float),
        ("mirostat_eta", ctypes.c_float),
        ("skip_special_token", ctypes.c_bool),
        ("is_async", ctypes.c_bool),
        ("img_start", ctypes.c_char_p), # For multimodal
        ("img_end", ctypes.c_char_p), # For multimodal
        ("img_content", ctypes.c_char_p), # For multimodal
        ("extend_param", RKLLMExtendParam),
    ]

class RKLLMResult(ctypes.Structure):
    """ Corresponds to RKLLMResult structure (Page 2-3) """
    _fields_ = [
        ("text", ctypes.c_char_p),
        ("token_id", ctypes.c_int32), 
    ]

class RKLLMEmbedInput(ctypes.Structure):
    _fields_ = [
        ("embed", ctypes.POINTER(ctypes.c_float)),
        ("n_tokens", ctypes.c_size_t)
    ]

class RKLLMTokenInput(ctypes.Structure):
    _fields_ = [
        ("input_ids", ctypes.POINTER(ctypes.c_int32)),
        ("n_tokens", ctypes.c_size_t)
    ]

class RKLLMMultiModelInput(ctypes.Structure):
    _fields_ = [
        ("prompt", ctypes.c_char_p),
        ("image_embed", ctypes.POINTER(ctypes.c_float)),
        ("n_image_tokens", ctypes.c_size_t),
        ("n_image", ctypes.c_size_t),
        ("image_width", ctypes.c_size_t),
        ("image_height", ctypes.c_size_t)
    ]

class RKLLMInputUnion(ctypes.Union):
    """ Union part of RKLLMInput (Page 7-8) """
    _fields_ = [
        ("prompt_input", ctypes.c_char_p),
    ]

class RKLLMInput(ctypes.Structure):
    """ Corresponds to RKLLMInput structure (Page 7-8) """
    _fields_ = [
        ("role", ctypes.c_char_p),
        ("enable_thinking", ctypes.c_bool),
        ("input_type", ctypes.c_int), 
        ("input_data", RKLLMInputUnion),
    ]

class RKLLMInferParam(ctypes.Structure):
    """ Corresponds to RKLLMInferParam structure (Page 10) """
    _fields_ = [
        ("mode", ctypes.c_int), 
        ("lora_params", ctypes.c_void_p), 
        ("prompt_cache_params", ctypes.c_void_p), 
        ("keep_history", ctypes.c_int), 
    ]

LLMResultCallback = ctypes.CFUNCTYPE(None, ctypes.POINTER(RKLLMResult), ctypes.c_void_p, ctypes.c_int)

""" LIBRARY LOADING AND INIT """

def find_library_path(lib_name: str) -> str:
    base = os.path.dirname(os.path.abspath(__file__))
    paths = ['.', base, os.path.join(base, 'lib'), '/usr/lib', '/usr/local/lib']
    if 'LD_LIBRARY_PATH' in os.environ:
        paths.extend(os.environ['LD_LIBRARY_PATH'].split(os.pathsep))
    for p in dict.fromkeys(paths):
        candidate = os.path.join(p, lib_name)
        if os.path.exists(candidate):
            return candidate
    return None

def init_rkllm_model() -> bool:
    global rkllm_lib, llm_handle, rkllm_params_global, model_initialized, conversation_history_bytes
    if model_initialized:
        return True
    lib_path = LIBRARY_PATH if os.path.exists(LIBRARY_PATH) else find_library_path('librkllmrt.so')
    if not lib_path:
        print("Error: RKLLM library not found.")
        return False
    rkllm_lib = ctypes.CDLL(lib_path)
    # Bind function signatures
    rkllm_lib.rkllm_createDefaultParam.restype = RKLLMParam
    rkllm_lib.rkllm_init.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(RKLLMParam), LLMResultCallback]
    rkllm_lib.rkllm_init.restype = ctypes.c_int
    rkllm_lib.rkllm_run.argtypes = [ctypes.c_void_p, ctypes.POINTER(RKLLMInput), ctypes.POINTER(RKLLMInferParam), ctypes.c_void_p]
    rkllm_lib.rkllm_run.restype = ctypes.c_int
    rkllm_lib.rkllm_clear_kv_cache.argtypes = [ctypes.c_void_p, ctypes.c_int]
    rkllm_lib.rkllm_clear_kv_cache.restype = ctypes.c_int
    rkllm_params_global = rkllm_lib.rkllm_createDefaultParam()
    
    # Configure params
    rkllm_params_global.model_path = MODEL_PATH.encode('utf-8')
    rkllm_params_global.max_context_len = MAX_CONTEXT_LENGTH
    rkllm_params_global.n_keep = N_KEEP
    if rkllm_params_global.max_new_tokens == 0 and MAX_NEW_TOKENS > 0:
        rkllm_params_global.max_new_tokens = MAX_NEW_TOKENS
    rkllm_params_global.use_gpu = USE_GPU
    rkllm_params_global.is_async = IS_ASYNC
    os.environ['RKLLM_LOG_LEVEL'] = str(LOG_LEVEL)

    ret = rkllm_lib.rkllm_init(ctypes.byref(llm_handle), ctypes.byref(rkllm_params_global), api_llm_callback)
    if ret != 0:
        print(f"Error: rkllm_init failed ({ret})")
        return False
    # Initialize history with default system prompt
    conversation_history_bytes = get_chat_template('default').replace(b"{system_message}", DEFAULT_SYSTEM_MESSAGE.encode('utf-8'))
    model_initialized = True
    return True


""" GET CHAT TEMPLATE AND DELIMITER TOKENS """

def get_chat_template(model_id: str) -> bytes:
    tpl = MODEL_TEMPLATES.get(model_id.lower())
    if not tpl:
        tpl = MODEL_TEMPLATES.get('default')
    return tpl.encode('utf-8')


""" Summarization Helpers """

def extract_turns(text: str) -> list:
    # split on plain-text role labels
    pattern = re.compile(r"(User: .*?\n|Assistant: .*?\n)", re.DOTALL)
    parts = pattern.findall(text)
    # group into turns
    turns = []
    for chunk in parts:
        turns.append(chunk)
    return turns
def summarize_old_turns(old_turns: list, model_id: str) -> str:
    prompt = get_chat_template(model_id).format(
        system_message="Summarize the following conversation concisely:",
        history_and_user_turn="".join(old_turns)
    )
    prompt_bytes = prompt.encode('utf-8')
    q = queue.Queue()
    # Map current thread to this queue for sync run
    tid = threading.get_ident()
    with thread_queue_map_lock:
        thread_queue_map[tid] = q
    # Prepare input
    rk_input = RKLLMInput()
    rk_input.input_type = RKLLM_INPUT_PROMPT
    rk_input.input_data.prompt_input = ctypes.c_char_p(prompt_bytes)
    infer = RKLLMInferParam()
    infer.mode = RKLLM_INFER_GENERATE
    infer.keep_history = 0
    # Run synchronously
    _ = rkllm_lib.rkllm_run(llm_handle, ctypes.byref(rk_input), ctypes.byref(infer), None)
    # Collect summary
    summary = ''
    while True:
        try:
            tok = q.get(timeout=5)
        except queue.Empty:
            break
        if tok is None or tok == '__LLM_RUN_FINISH__':
            break
        summary += tok
    # Clean up mapping
    with thread_queue_map_lock:
        thread_queue_map.pop(tid, None)
    return summary

@LLMResultCallback
def api_llm_callback(ptr, userdata, state):
    tid = threading.get_ident()
    with thread_queue_map_lock:
        q = thread_queue_map.get(tid)
    if not q:
        return
    if not ptr:
        q.put(None)
        return
    res = ptr.contents
    if res.token_id in EOS_TOKEN_IDS:
        q.put("__LLM_RUN_FINISH__")
        return
    if state == LLM_RUN_NORMAL and res.text:
        txt = res.text.decode('utf-8', errors='replace')
        q.put(txt)
    elif state == LLM_RUN_FINISH:
        q.put("__LLM_RUN_FINISH__")
    else:
        q.put(None)
        
def rkllm_inference_worker(handle, input_obj_ptr, infer_params_obj_ptr, q: queue.Queue):
    tid = threading.get_ident()
    with thread_queue_map_lock:
        thread_queue_map[tid] = q
    try:
        run_ret = rkllm_lib.rkllm_run(handle, input_obj_ptr, infer_params_obj_ptr, None)
        if run_ret != 0:
            q.put(None)
    finally:
        with thread_queue_map_lock:
            thread_queue_map.pop(tid, None)
            
def format_openai_error_response(message, err_type="invalid_request_error", param=None, code=None):
    error_obj = {"error": {"message": message, "type": err_type}}
    if param: error_obj["error"]["param"] = param
    if code: error_obj["error"]["code"] = code
    return jsonify(error_obj)

@app.route('/v1/models', methods=['GET'])
def list_models():
    """Return available models with their capabilities"""
    if not model_initialized:
        return format_openai_error_response("RKLLM model not initialized.", "api_error"), 500
    model_name = rkllm_params_global.model_path.decode().split('/')[-1] if rkllm_params_global else "rkllm_model"
    models = {
        "object": "list",
        "data": [
            {
                "id": model_name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "user",
                "permission": [],
                "root": model_name,
                "parent": None,
                # "capabilities": {
                #     "tool_calling": True  # Indicate that this model supports tool calling
                # }
            }
        ]
    }
    return jsonify(models)

@app.route('/v1/stop', methods=['POST'])
def stop_generation():
    data = request.get_json() or {}
    req_id = data.get('request_id')
    if not req_id:
        return jsonify({'error':'request_id required'}), 400
    with active_streams_lock:
        ev = active_streams.pop(req_id, None)
    if ev:
        ev.set()
        rkllm_lib.rkllm_clear_kv_cache(llm_handle, 1)
        return jsonify({'stopped':True}), 200
    return jsonify({'error':'stream not found'}), 404

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions_handler():
    
    #request parsing
    
    global conversation_history
    if not model_initialized:
        return jsonify({'error':'model not initialized'}), 500
    data = request.get_json(force=True)
    messages = data.get('messages', [])
    stream_mode = data.get('stream', False)
    model_id = data.get('model', 'default')
    if not messages or messages[-1].get('role') != 'user':
        return jsonify({'error':"Last message must be 'user'"}), 400
    user_text = messages[-1]['content']
    
    # Summarize history if above 80 percent of max context length
    with conversation_lock:
        est_tokens = len(conversation_history) // 4
        threshold = int(MAX_CONTEXT_LENGTH * 0.8)
        if est_tokens > threshold:
            turns = extract_turns(conversation_history)
            old, recent = turns[:-6], turns[-6:]
            summary = summarize_old_turns(old, model_id)
            conversation_history = summary + ''.join(recent)
        # Append new user turn
        conversation_history += f"User: {user_text}\n"
        
    # Build prompt
    tpl = get_chat_template(model_id)
    prompt_str = tpl.format(
        system_message= DEFAULT_SYSTEM_MESSAGE,
        history_and_user_turn=conversation_history
    )
    prompt_bytes = prompt_str.encode('utf-8')
    
    # Prepare RKLLM input objects
    rk_input = RKLLMInput()
    rk_input.input_type = RKLLM_INPUT_PROMPT
    rk_input.input_data.prompt_input = ctypes.c_char_p(prompt_bytes)
    infer = RKLLMInferParam()
    infer.mode = RKLLM_INFER_GENERATE
    infer.keep_history = 1
    req_id = f"chatcmpl-{uuid.uuid4().hex}"
    model_name = os.path.basename(rkllm_params_global.model_path.decode())
    created_ts = int(time.time())
    
    # Create a fresh queue for this request
    q = queue.Queue()
    if stream_mode:
        stop_ev = threading.Event()
        with active_streams_lock:
            active_streams[req_id] = stop_ev
        # Start worker thread
        worker = threading.Thread(
            target=rkllm_inference_worker,
            args=(llm_handle, ctypes.byref(rk_input), ctypes.byref(infer), q),
            daemon=True
        )
        worker.start()

        def event_stream():
            first = True
            assistant_resp = ''
            while True:
                if stop_ev.is_set():
                    break
                try:
                    tok = q.get(timeout=30)
                except queue.Empty:
                    break
                if tok is None or tok == '__LLM_RUN_FINISH__':
                    break
                assistant_resp += tok
                if first:
                    chunk = {
                        'id': req_id,
                        'object': 'chat.completion.chunk',
                        'created': created_ts,
                        'model': model_name,
                        'choices': [{
                            'index': 0,
                            'delta': {'role': 'assistant', 'content': tok},
                            'finish_reason': None
                        }]
                    }
                    first = False
                else:
                    chunk = {
                        'id': req_id,
                        'object': 'chat.completion.chunk',
                        'created': created_ts,
                        'model': model_name,
                        'choices': [{
                            'index': 0,
                            'delta': {'content': tok},
                            'finish_reason': None
                        }]
                    }
                yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"
            with active_streams_lock:
                active_streams.pop(req_id, None)
            with conversation_lock:
                conversation_history += f"Assistant: {assistant_resp}\n"

        return Response(stream_with_context(event_stream()), mimetype='text/event-stream')

    # One-shot
    # Map current thread for callback
    tid = threading.get_ident()
    with thread_queue_map_lock:
        thread_queue_map[tid] = q
    run_ret = rkllm_lib.rkllm_run(llm_handle, ctypes.byref(rk_input), ctypes.byref(infer), None)
    # Clean up mapping
    with thread_queue_map_lock:
        thread_queue_map.pop(tid, None)
    if run_ret != 0:
        return jsonify({'error': f'rkllm_run failed ({run_ret})'}), 500
    assistant_resp = ''
    while True:
        try:
            tok = q.get(timeout=30)
        except queue.Empty:
            break
        if tok is None or tok == '__LLM_RUN_FINISH__':
            break
        assistant_resp += tok
    with conversation_lock:
        conversation_history += f"Assistant: {assistant_resp}\n"
    response = {
        'id': req_id,
        'object': 'chat.completion',
        'created': created_ts,
        'model': model_name,
        'choices': [{
            'index': 0,
            'message': {'role': 'assistant', 'content': assistant_resp},
            'finish_reason': 'stop'
        }]
    }
    return jsonify(response)

# --- Main ---
if __name__ == '__main__':
    if init_rkllm_model():
        app.run(host=SERVER_HOST, port=SERVER_PORT, threaded=True, debug=DEBUG_MODE)
    else:
        print("Failed to initialize RKLLM model.")


# --- Tool Calling Utilities ---
# def try_parse_tool_calls(content):
#     """
#     Parse tool calls from model output using regex pattern for Qwen3 format.
#     Returns a dictionary with parsed tool calls if found.
#     """
#     try:
#         tool_calls = []
#         offset = 0
        
#         # Look for tool calls in the <tool_call>...</tool_call> format
#         for i, m in enumerate(re.finditer(r"<tool_call>\n(.+?)?\n</tool_call>", content, re.DOTALL)):
#             if i == 0:
#                 offset = m.start()
            
#             try:
#                 # Parse the JSON content inside tool_call tags
#                 func_text = m.group(1).strip()
#                 func = json.loads(func_text)
                
#                 # Ensure the function has required fields
#                 if not func.get("name"):
#                     print(f"Warning: Tool call missing 'name' field: {func_text}")
#                     continue
                    
#                 if isinstance(func.get("arguments", ""), str):
#                     # Convert arguments string to dict if it's a valid JSON string
#                     try:
#                         args_text = func["arguments"].strip()
#                         # Handle empty arguments
#                         if not args_text:
#                             func["arguments"] = {}
#                         else:
#                             try:
#                                 func["arguments"] = json.loads(args_text)
#                             except json.JSONDecodeError:
#                                 # If not valid JSON, keep as string but log warning
#                                 print(f"Warning: Invalid JSON in arguments: {args_text}")
#                     except Exception as e:
#                         print(f"Error processing arguments: {e}")
#                         func["arguments"] = {}
                        
#                 tool_calls.append({
#                     "index": i,
#                     "id": f"call_{uuid.uuid4().hex[:8]}",
#                     "type": "function", 
#                     "function": {
#                         "name": func.get("name", ""),
#                         "arguments": func.get("arguments", {})
#                     }
#                 })
#             except json.JSONDecodeError as e:
#                 print(f"Failed to parse tool call: {m.group(1)}\nError: {e}")
#             except Exception as e:
#                 print(f"Unexpected error parsing tool call: {e}")
        
#         # If tool calls were found
#         if tool_calls:
#             # Extract any content before the first tool call
#             content_before_tools = ""
#             if offset > 0 and content[:offset].strip():
#                 content_before_tools = content[:offset].strip()
            
#             return {"content": content_before_tools, "tool_calls": tool_calls}
        
#         # If no tool calls, just return the cleaned content
#         return {"content": re.sub(r"<\|im_end\|>$", "", content)}
#     except Exception as e:
#         print(f"Error in try_parse_tool_calls: {e}")
#         # In case of any error, return the original content to avoid crashes
#         return {"content": content}

# def format_tool_calls_for_response(tool_calls):
#     """
#     Format tool calls for the OpenAI API response format
#     """
#     formatted_tool_calls = []
    
#     for call in tool_calls:
#         formatted_call = {
#             "id": call["id"],
#             "type": call["type"],
#             "function": {
#                 "name": call["function"]["name"],
#                 "arguments": json.dumps(call["function"]["arguments"])
#             }
#         }
#         formatted_tool_calls.append(formatted_call)
    
#     return formatted_tool_calls






# --- Worker function to run RKLLM inference in a separate thread ---
# def rkllm_inference_worker(handle, input_obj_ptr, infer_params_obj_ptr):
#     """
#     This function will be run in a separate thread for streaming requests.
#     It calls the blocking rkllm_run. Callbacks will populate the queue.
#     """
#     thread_id = threading.get_ident()
#     print(f"[Worker {thread_id}] Starting rkllm_run. Timestamp: {time.time()}")
#     run_ret = rkllm_lib.rkllm_run(handle, input_obj_ptr, infer_params_obj_ptr, None)
#     print(f"[Worker {thread_id}] rkllm_run finished with code: {run_ret}. Timestamp: {time.time()}")
#     if run_ret != 0:
#         # If run fails, callback might not send FINISH. Put an error signal.
#         print(f"[Worker {thread_id}] rkllm_run error, ensuring error signal in queue.")
#         stream_token_queue.put(None) # Signal error to streamer
#     # The __LLM_RUN_FINISH__ or None (for error) should be put by the callback itself.
#     # This function just ensures rkllm_run is called.

# @app.route('/v1/chat/completions', methods=['POST'])
# def chat_completions_handler():
#     global conversation_history_bytes, current_assistant_response_parts, current_tool_calls, rkllm_params_global, llm_handle, rkllm_lib, stream_token_queue
    
#     if not model_initialized: return format_openai_error_response("RKLLM model not initialized.", "api_error"), 500
#     try: request_data = request.get_json()
#     except Exception as e: return format_openai_error_response(f"Invalid JSON: {e}", "invalid_request_error"), 400
#     if not request_data: return format_openai_error_response("Request body missing or not JSON.", "invalid_request_error"), 400

#     messages = request_data.get("messages", [])
#     stream_requested = request_data.get("stream", False)
    
#     # Extract tool/function data from the request
#     tools = request_data.get("tools", [])
#     functions = request_data.get("functions", [])
#     # If both tools and functions provided, tools take precedence per OpenAI API specification
#     function_call = request_data.get("function_call", None)
#     tool_choice = request_data.get("tool_choice", None)
    
#     # Convert functions to tools format if provided (backward compatibility)
#     if not tools and functions:
#         tools = [{"type": "function", "function": func} for func in functions]
    
#     if not messages or messages[-1].get("role") != "user": return format_openai_error_response("Last message must be 'user'.", "invalid_request_error", param="messages"), 400
#     last_user_message_content = messages[-1].get("content", "")
#     if not last_user_message_content.strip(): return format_openai_error_response("User content is empty.", "invalid_request_error", param="messages.content"), 400

#     # rkllm_lock ensures that context management, rkllm_run call, and history updates are atomic per request.
#     with rkllm_lock: 
#         # Clear any previous tool calls
#         current_tool_calls.clear()
        
#         # Format the user message
#         current_user_turn_bytes = USER_TURN_START + last_user_message_content.encode('utf-8') + USER_TURN_END

#         # Initialize system message before any modifications
#         system_message = DEFAULT_SYSTEM_MESSAGE

#         # Custom system message in request
#         if "system" in request_data and request_data["system"]:
#             system_message = request_data["system"]

#         # If tools are provided, add them to the system prompt
#         if tools or functions:
#             try:
#                 # Prefer tools over functions if both are provided (tools is the newer format)
#                 tools_to_use = tools if tools else []
                
#                 # Convert functions to tools if provided (backward compatibility)
#                 if functions:
#                     for func in functions:
#                         tools_to_use.append({
#                             "type": "function",
#                             "function": func
#                         })
                
#                 # Create a more structured but simpler tools description for the model
#                 tools_info = "Available tools:\n"
#                 for i, tool in enumerate(tools_to_use):
#                     try:
#                         if tool["type"] == "function":
#                             func = tool["function"]
#                             func_name = func.get("name", f"unnamed_function_{i}")
#                             func_desc = func.get("description", "No description provided")
                            
#                             # Add function name and description
#                             tools_info += f"\n{i+1}. {func_name}: {func_desc}"
                            
#                             # Add parameters in a simpler format
#                             if "parameters" in func:
#                                 tools_info += "\n   Parameters:"
                                
#                                 # Add required parameters if specified
#                                 if "required" in func["parameters"]:
#                                     tools_info += f"\n   - Required: {', '.join(func['parameters'].get('required', []))}"
                                
#                                 # Add properties in a simplified way
#                                 if "properties" in func["parameters"]:
#                                     for param_name, param in func["parameters"]["properties"].items():
#                                         param_type = param.get("type", "any")
#                                         param_desc = param.get("description", "")
#                                         tools_info += f"\n   - {param_name} ({param_type}): {param_desc}"
#                     except Exception as e:
#                         print(f"Error processing tool {i}: {e}")
#                         continue

#                 # Add tools information to the system message
#                 tools_info += "\n\nWhen using a tool, use the exact format:\n<tool_call>\n{\"name\": \"function_name\", \"arguments\": {\"param1\": \"value1\", \"param2\": \"value2\"}}\n</tool_call>\n"
                
#                 # Append tools info to system message
#                 system_message = system_message + "\n\n" + tools_info

#                 # Handle tool_choice if provided (simplified)
#                 if "tool_choice" in request_data or "function_call" in request_data:
#                     # Prefer tool_choice over function_call if both are provided
#                     tool_choice = request_data.get("tool_choice", request_data.get("function_call", None))
#                     if tool_choice:
#                         # Extract just the function name for a simpler instruction
#                         if isinstance(tool_choice, dict):
#                             if tool_choice.get("type") == "function" and "function" in tool_choice:
#                                 func_name = tool_choice["function"].get("name", "")
#                                 if func_name:
#                                     system_message += f"\n\nYou MUST use the {func_name} tool for this request."
#             except Exception as e:
#                 print(f"Error adding tools to system prompt: {e}")
#                 # Continue without tools if there's an error
        
#         # Format system prompt
#         if system_message:
#             system_prompt_bytes = SYS_PROMPT_TEMPLATE.replace(b"{system_message}", system_message.encode('utf-8'))
#             conversation_history_bytes = system_prompt_bytes + conversation_history_bytes
        
#         prompt_for_llm = conversation_history_bytes + current_user_turn_bytes + ASSISTANT_TURN_START
        
#         EFFECTIVE_PROMPT_TOKEN_LIMIT = 500 
#         estimated_prompt_tokens = len(prompt_for_llm) / 3.5
        
#         if estimated_prompt_tokens > EFFECTIVE_PROMPT_TOKEN_LIMIT:
#             print(f"\n[API Ctx Mgmt] Est. prompt tokens ({int(estimated_prompt_tokens)}) > limit ({EFFECTIVE_PROMPT_TOKEN_LIMIT}). Clearing KV cache.")
#             clear_ret = rkllm_lib.rkllm_clear_kv_cache(llm_handle, 1) 
#             if clear_ret == 0:
#                 print("[API Ctx Mgmt] KV cache cleared.")
#                 conversation_history_bytes = SYS_PROMPT_TEMPLATE.replace(b"{system_message}", DEFAULT_SYSTEM_MESSAGE.encode('utf-8'))
#                 prompt_for_llm = conversation_history_bytes + current_user_turn_bytes + ASSISTANT_TURN_START
#                 print(f"[API Ctx Mgmt] History reset. New prompt est: {int(len(prompt_for_llm)/3.5)} tokens.")
#             else:
#                 print(f"[API Ctx Mgmt] Error clearing KV cache (code: {clear_ret}).")
        
#         current_assistant_response_parts.clear()
#         while not stream_token_queue.empty():
#             try: stream_token_queue.get_nowait()
#             except queue.Empty: break
        
#         # These objects are passed to the C function, ensure they exist for the duration of the call.
#         # For threaded calls, they must persist until the thread is done.
#         # ctypes.byref creates a light-weight pointer. The original Python objects must be kept alive.
#         rkllm_input_obj = RKLLMInput() 
#         rkllm_input_obj.input_type = RKLLM_INPUT_PROMPT
#         rkllm_input_obj.prompt_input = ctypes.c_char_p(prompt_for_llm)
        
#         rkllm_infer_params_obj = RKLLMInferParam()
#         rkllm_infer_params_obj.mode = RKLLM_INFER_GENERATE
#         rkllm_infer_params_obj.keep_history = 1
        
#         request_id = f"chatcmpl-{uuid.uuid4().hex}"
#         model_name_to_return = rkllm_params_global.model_path.decode().split('/')[-1] if rkllm_params_global else "rkllm_model"
#         created_ts = int(time.time())
        
#         run_ret_holder = {"value": 0} # To get return value from worker thread if needed, not strictly used now

#         if stream_requested:
#             # For streaming, run rkllm_run in a separate thread.
#             # The main Flask thread will return the generator that pulls from the queue.
#             print(f"[API Handler {threading.get_ident()}] Starting worker thread for rkllm_run. Timestamp: {time.time()}")
            
#             # Keep references to pass to thread
#             input_obj_ptr = ctypes.byref(rkllm_input_obj)
#             infer_params_obj_ptr = ctypes.byref(rkllm_infer_params_obj)

#             worker_thread = threading.Thread(target=rkllm_inference_worker, 
#                                              args=(llm_handle, input_obj_ptr, infer_params_obj_ptr))
#             worker_thread.start()
#             # The main thread does not wait for worker_thread to finish here.
#             # It proceeds to return the streaming response.

#             def generate_stream_response_sse():
#                 global conversation_history_bytes, current_tool_calls # To update after stream
                
#                 print(f"[Stream SSE Gen {threading.get_ident()}] Starting generator. Timestamp: {time.time()}")
#                 streamed_response_for_history_update = [] 
#                 accumulated_text = ""
#                 tool_call_detected = False
#                 first_content_chunk_sent = False
                
#                 try:
#                     while True:
#                         try:
#                             token_or_signal = stream_token_queue.get(timeout=45.0) # Increased timeout
#                         except queue.Empty:
#                             print(f"[Stream SSE Gen {threading.get_ident()}] Timeout waiting for item from queue. Timestamp: {time.time()}")
#                             break 

#                         if token_or_signal is None: 
#                             print(f"[Stream SSE Gen {threading.get_ident()}] Received None (error/end signal). Timestamp: {time.time()}")
#                             break 
                        
#                         if token_or_signal == "__LLM_RUN_FINISH__":
#                             print(f"[Stream SSE Gen {threading.get_ident()}] Received FINISH signal. Timestamp: {time.time()}")
                            
#                             # Check the accumulated text for tool calls at the end
#                             if accumulated_text:
#                                 parsed_response = try_parse_tool_calls(accumulated_text)
#                                 if "tool_calls" in parsed_response:
#                                     # We found tool calls in the final response
#                                     tool_call_detected = True
#                                     current_tool_calls = parsed_response["tool_calls"]
                                    
#                                     # Send the tool calls as deltas
#                                     for i, tool_call in enumerate(current_tool_calls):
#                                         formatted_call = {
#                                             "id": tool_call["id"],
#                                             "type": tool_call["type"],
#                                             "function": {
#                                                 "name": tool_call["function"]["name"],
#                                                 "arguments": json.dumps(tool_call["function"]["arguments"])
#                                             }
#                                         }
                                        
#                                         # For the first tool call, set content to empty string if we haven't sent content yet
#                                         if i == 0 and not first_content_chunk_sent:
#                                             chunk = {
#                                                 "id": request_id, 
#                                                 "object": "chat.completion.chunk", 
#                                                 "created": created_ts, 
#                                                 "model": model_name_to_return, 
#                                                 "choices": [
#                                                     {
#                                                         "index": 0, 
#                                                         "delta": {
#                                                             "role": "assistant",
#                                                             "content": parsed_response.get("content", "")
#                                                         },
#                                                         "finish_reason": None
#                                                     }
#                                                 ]
#                                             }
#                                             yield f"data: {json.dumps(chunk)}\n\n"
#                                             first_content_chunk_sent = True
                                        
#                                         # Send the tool call as a delta
#                                         chunk = {
#                                             "id": request_id, 
#                                             "object": "chat.completion.chunk", 
#                                             "created": created_ts, 
#                                             "model": model_name_to_return, 
#                                             "choices": [
#                                                 {
#                                                     "index": 0, 
#                                                     "delta": {"tool_calls": [formatted_call]},
#                                                     "finish_reason": None
#                                                 }
#                                             ]
#                                         }
#                                         yield f"data: {json.dumps(chunk)}\n\n"
                            
#                             # Send the final chunk with the appropriate finish_reason
#                             finish_reason = "tool_calls" if tool_call_detected else "stop"
#                             final_chunk = {"id": request_id, "object": "chat.completion.chunk", "created": created_ts, "model": model_name_to_return, "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}]}
#                             yield f"data: {json.dumps(final_chunk)}\n\n"
#                             break
                        
#                         streamed_response_for_history_update.append(token_or_signal) 
#                         accumulated_text += token_or_signal
                        
#                         # Only send content chunks if we haven't detected a tool call pattern yet
#                         if not tool_call_detected:
#                             # Check if we might have a complete tool call
#                             if "<tool_call>" in accumulated_text and "</tool_call>" in accumulated_text:
#                                 # Potential tool call detected - parse the current text
#                                 parsed = try_parse_tool_calls(accumulated_text)
#                                 if "tool_calls" in parsed:
#                                     # We found tool calls - don't send more content after this
#                                     tool_call_detected = True
                            
#                             # If no tool call detected yet, stream the token as content
#                             if not tool_call_detected:
#                                 if not first_content_chunk_sent:
#                                     # First chunk needs to include the role
#                                     chunk = {
#                                         "id": request_id, 
#                                         "object": "chat.completion.chunk", 
#                                         "created": created_ts, 
#                                         "model": model_name_to_return, 
#                                         "choices": [
#                                             {
#                                                 "index": 0, 
#                                                 "delta": {"role": "assistant", "content": token_or_signal},
#                                                 "finish_reason": None
#                                             }
#                                         ]
#                                     }
#                                     first_content_chunk_sent = True
#                                 else:
#                                     chunk = {
#                                         "id": request_id, 
#                                         "object": "chat.completion.chunk", 
#                                         "created": created_ts, 
#                                         "model": model_name_to_return, 
#                                         "choices": [
#                                             {
#                                                 "index": 0, 
#                                                 "delta": {"content": token_or_signal},
#                                                 "finish_reason": None
#                                             }
#                                         ]
#                                     }
#                                 yield f"data: {json.dumps(chunk)}\n\n"
#                         # time.sleep(0.005) # Minimal sleep to allow other threads, if any, to process. Can make streaming more "visible".
                
#                 except Exception as e_stream:
#                     print(f"[Stream SSE Gen {threading.get_ident()}] Error: {e_stream}. Timestamp: {time.time()}")
#                 finally:
#                     print(f"[Stream SSE Gen {threading.get_ident()}] Sending [DONE]. Timestamp: {time.time()}")
#                     yield f"data: [DONE]\n\n"
                    
#                     # Update global conversation history after stream is complete.
#                     # This needs to be done carefully with the rkllm_lock.
#                     # The worker thread might still be running or just finished.
#                     # The lock is acquired by the main request handler.
#                     # This update should ideally happen once the worker thread confirms completion.
#                     # For now, we assume current_assistant_response_parts got populated by the callbacks
#                     # which were triggered by the worker thread.
                    
#                     # The lock is held by the parent (chat_completions_handler)
#                     # We need to ensure the worker_thread has finished before updating history based on its work.
#                     # However, we can't block the stream yield for worker_thread.join().
#                     # The current_assistant_response_parts is filled by the callback.
#                     # If rkllm_run was successful (worker didn't crash), current_assistant_response_parts should be accurate.
                    
#                     # Let's wait for the worker thread here before updating history.
#                     # This means the [DONE] signal might be delayed until the worker thread fully exits.
#                     print(f"[Stream SSE Gen {threading.get_ident()}] Waiting for worker thread to complete before history update. Timestamp: {time.time()}")
#                     worker_thread.join(timeout=5.0) # Wait for worker to finish
#                     if worker_thread.is_alive():
#                         print(f"[Stream SSE Gen {threading.get_ident()}] WARNING: Worker thread did not complete in time after stream.")

#                     # Now that worker is done (or timed out), current_assistant_response_parts should be stable.
#                     # The lock is still held by the main request handler.
#                     full_response_this_turn = "".join(current_assistant_response_parts) # These parts were for this turn
#                     if full_response_this_turn:
#                         # Check if there were any tool calls in the response
#                         if current_tool_calls:
#                             # We had tool calls - only store the content part in history
#                             parsed_response = try_parse_tool_calls(full_response_this_turn)
#                             # Extract only the content part (without the tool calls)
#                             content_for_history = parsed_response.get("content", "")
#                             new_segment = current_user_turn_bytes + ASSISTANT_TURN_START + content_for_history.encode('utf-8') + ASSISTANT_TURN_END
#                         else:
#                             # No tool calls - store the full response
#                             new_segment = current_user_turn_bytes + ASSISTANT_TURN_START + full_response_this_turn.encode('utf-8') + ASSISTANT_TURN_END
                        
#                         conversation_history_bytes += new_segment
#                         print(f"[API History Update after stream generation] History bytes: {len(conversation_history_bytes)}, Had tool calls: {bool(current_tool_calls)}")
                        
#                         # Clear the tool calls list after processing
#                         current_tool_calls.clear()


#             return Response(stream_with_context(generate_stream_response_sse()), mimetype='text/event-stream')
        
#         else: # One-shot response (rkllm_run is blocking if is_async=False)
#             print(f"[API Handler {threading.get_ident()}] Calling rkllm_run (sync). Timestamp: {time.time()}")
#             run_start_time = time.time()
#             run_ret = rkllm_lib.rkllm_run(llm_handle, ctypes.byref(rkllm_input_obj), ctypes.byref(rkllm_infer_params_obj), None)
#             run_end_time = time.time()
#             print(f"[API Handler {threading.get_ident()}] rkllm_run (sync) returned: {run_ret}. Duration: {run_end_time - run_start_time:.4f}s. Qsize: {stream_token_queue.qsize()}. Timestamp: {time.time()}")

#             if run_ret != 0:
#                 print(f"[API Error] rkllm_run (sync) failed (code {run_ret}).")
#                 return format_openai_error_response(f"RKLLM run failed (code: {run_ret}).", "api_error"), 500

#             full_assistant_response_str = "".join(current_assistant_response_parts)
            
#             # For non-streaming mode, we collected the complete assistant response in current_assistant_response_parts
#             try:
#                 full_response = "".join(current_assistant_response_parts)
                
#                 # Check for tool calls in response
#                 parsed_response = try_parse_tool_calls(full_response)
                
#                 # Create the chat completion message from assistant
#                 assistant_message = {
#                     "role": "assistant",
#                     "content": parsed_response.get("content", full_response)
#                 }
                
#                 # Add tool calls to message if present
#                 finish_reason = "stop"
#                 if "tool_calls" in parsed_response and parsed_response["tool_calls"]:
#                     try:
#                         current_tool_calls = parsed_response["tool_calls"] # Store for potential follow-up
#                         formatted_calls = format_tool_calls_for_response(parsed_response["tool_calls"])
#                         assistant_message["tool_calls"] = formatted_calls
#                         finish_reason = "tool_calls"
                        
#                         # Debug
#                         print(f"Found tool calls: {len(current_tool_calls)}")
#                         for i, tool_call in enumerate(current_tool_calls):
#                             print(f"Tool call {i+1}: {tool_call['function'].get('name', 'unknown')}")
#                     except Exception as e:
#                         print(f"Error processing tool calls: {e}")
#             except Exception as e:
#                 print(f"Error processing non-streaming response: {e}")
#                 assistant_message = {
#                     "role": "assistant", 
#                     "content": full_response
#                 }
#                 finish_reason = "stop"
                
#             # Update conversation history
#             if run_ret == 0:
#                 try:
#                     # Only store the content part in history, without tool calls JSON
#                     if "tool_calls" in parsed_response and parsed_response["tool_calls"]:
#                         # Add just the text content to the history, not the tool call JSON
#                         response_for_history = parsed_response.get("content", "")
#                     else:
#                         # No tool calls, use the full response
#                         response_for_history = full_response
                        
#                     # Add this turn to conversation history
#                     new_segment = current_user_turn_bytes + ASSISTANT_TURN_START + response_for_history.encode('utf-8') + ASSISTANT_TURN_END
#                     conversation_history_bytes += new_segment
#                     print(f"[API History Update] History bytes: {len(conversation_history_bytes)}")
#                 except Exception as e:
#                     print(f"Error updating history: {e}")
            
#             # Prepare the API response
#             prompt_tokens_est = len(prompt_for_llm) // 4 
#             completion_tokens_est = len(full_response) // 4
                
#             # Assemble the final API response
#             response_json = {
#                 "id": request_id, 
#                 "object": "chat.completion", 
#                 "created": created_ts, 
#                 "model": model_name_to_return, 
#                 "choices": [{"index": 0, "message": assistant_message, "finish_reason": finish_reason}],
#                 "usage": {"prompt_tokens": prompt_tokens_est, "completion_tokens": completion_tokens_est, "total_tokens": prompt_tokens_est + completion_tokens_est}
#             }
#             return jsonify(response_json)

# # --- Main Entry Point ---
# if __name__ == '__main__':
#     if init_rkllm_model():
#         print(f"Starting Flask server for RKLLM OpenAI-compliant API on http://{SERVER_HOST}:{SERVER_PORT}{API_BASE_PATH}/chat/completions")
#         app.run(host=SERVER_HOST, port=SERVER_PORT, threaded=True, debug=DEBUG_MODE) 
#     else:
#         print("Failed to initialize RKLLM model. Server not starting.")
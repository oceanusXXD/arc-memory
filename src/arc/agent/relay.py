"""Small Anthropic Messages relay for Claude Code and SiliconFlow."""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def normalize_request(body: dict, *, temperature: float, max_tokens: int,
                      thinking: bool = True, thinking_budget_tokens: int = 1024) -> dict:
    body = dict(body)
    system = body.get('system') or []
    system = list(system) if isinstance(system, list) else [{'type': 'text', 'text': str(system)}]
    messages = []
    for message in body.get('messages', []):
        if message.get('role') == 'system':
            content = message.get('content', '')
            system.extend(content if isinstance(content, list) else [{'type': 'text', 'text': str(content)}])
        else:
            messages.append(message)
    body.update(messages=messages, temperature=temperature, max_tokens=max_tokens)
    # Keep reasoning explicit for Qwen.  Claude Code remains the only client;
    # this relay only translates the Anthropic Messages request for SiliconFlow.
    body['thinking'] = ({'type': 'enabled', 'budget_tokens': int(thinking_budget_tokens)}
                        if thinking else {'type': 'disabled'})
    if system:
        body['system'] = system
    # Anthropic adaptive reasoning controls do not define Qwen sampling.
    body.pop('output_config', None)
    return body


class MessagesRelay:
    def __init__(self, upstream: str, *, temperature: float, max_tokens: int,
                 thinking: bool = True, thinking_budget_tokens: int = 1024,
                 timeout: float = 120):
        self.requests: list[dict] = []
        relay = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_POST(self):
                route = self.path.split('?', 1)[0].rstrip('/')
                if route not in {'/v1/messages', '/v1/messages/count_tokens'}:
                    self.send_error(404)
                    return
                try:
                    body = json.loads(self.rfile.read(int(self.headers.get('content-length', '0'))))
                    if route == '/v1/messages':
                        body = normalize_request(
                            body, temperature=temperature, max_tokens=max_tokens,
                            thinking=thinking, thinking_budget_tokens=thinking_budget_tokens,
                        )
                    record = {'path': route, 'request': body}
                    relay.requests.append(record)
                    headers = {k: v for k, v in self.headers.items()
                               if k.lower() not in {'host', 'content-length', 'connection', 'accept-encoding'}}
                    request = Request(upstream.rstrip('/') + self.path,
                                      data=json.dumps(body, ensure_ascii=False).encode(), headers=headers, method='POST')
                    try:
                        response = urlopen(request, timeout=timeout)
                    except HTTPError as exc:
                        response = exc
                    with response:
                        data = response.read()
                        record['status'] = response.status
                        record['response'] = data.decode('utf-8', errors='replace')
                        self.send_response(response.status)
                        self.send_header('content-type', response.headers.get('content-type', 'application/json'))
                        self.send_header('content-length', str(len(data)))
                        self.end_headers()
                        self.wfile.write(data)
                except Exception as exc:
                    data = json.dumps({'type': 'error', 'error': {'type': 'api_error', 'message': str(exc)}}).encode()
                    try:
                        self.send_response(502)
                        self.send_header('content-type', 'application/json')
                        self.send_header('content-length', str(len(data)))
                        self.end_headers()
                        self.wfile.write(data)
                    except (BrokenPipeError, ConnectionResetError):
                        pass

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        return f'http://127.0.0.1:{self.server.server_port}'

    def start(self):
        self.thread.start()
        return self

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

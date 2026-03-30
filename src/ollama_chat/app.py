# Licensed under the MIT License
# https://github.com/craigahobbs/ollama-chat/blob/main/LICENSE

"""
The ollama-chat back-end application
"""

from contextlib import contextmanager
import copy
import ctypes
import json
import os
from functools import partial
import platform
import importlib.resources
import re
import threading
import uuid

import chisel
import urllib3
import schema_markdown

from .chat import ChatManager, config_conversation, config_template_prompts
from .ollama import ollama_delete, ollama_list, ollama_pull


# The ollama-chat back-end API WSGI application class
class OllamaChat(chisel.Application):
    __slots__ = ('config', 'xorigin', 'chats', 'downloads', 'pool_manager')


    def __init__(self, config_path, xorigin=False):
        super().__init__()
        self.config = ConfigManager(config_path)
        self.xorigin = xorigin
        self.chats = {}
        self.downloads = {}
        self.pool_manager = urllib3.PoolManager(num_pools=10, maxsize=10)

        # Back-end documentation
        self.add_requests(chisel.create_doc_requests())

        # Back-end APIs
        self.add_request(create_template)
        self.add_request(delete_conversation)
        self.add_request(delete_conversation_exchange)
        self.add_request(delete_model)
        self.add_request(delete_template)
        self.add_request(download_model)
        self.add_request(get_conversation)
        self.add_request(get_conversations)
        self.add_request(get_models)
        self.add_request(get_system_info)
        self.add_request(get_template)
        self.add_request(move_conversation)
        self.add_request(move_template)
        self.add_request(regenerate_conversation_exchange)
        self.add_request(reply_conversation)
        self.add_request(set_conversation_thinking)
        self.add_request(set_conversation_title)
        self.add_request(check_grammar)
        self.add_request(clear_model)
        self.add_request(execute_curl)
        self.add_request(fetch_url)
        self.add_request(read_directory)
        self.add_request(translate_text)
        self.add_request(get_model_info)
        self.add_request(set_model)
        self.add_request(set_model_options)
        self.add_request(start_conversation)
        self.add_request(start_template)
        self.add_request(stop_conversation)
        self.add_request(stop_model_download)
        self.add_request(update_template)

        # Front-end statics
        self.add_static('index.html', urls=(('GET', None), ('GET', '/')))
        self.add_static('ollamaChat.bare')
        self.add_static('ollamaChatConversation.bare')
        self.add_static('ollamaChatModels.bare')
        self.add_static('ollamaChatTemplate.bare')
        self.add_static('ollamaChatUtil.bare')


    def add_static(self, filename, urls=(('GET', None),), doc_group='Ollama Chat Statics'):
        content_type = _CONTENT_TYPES.get(os.path.splitext(filename)[1], 'text/plain; charset=utf-8')
        with importlib.resources.files('ollama_chat.static').joinpath(filename).open('rb') as fh:
            self.add_request(chisel.StaticRequest(filename, fh.read(), content_type, urls, doc_group=doc_group))


    def __call__(self, environ, start_response):
        if self.xorigin:
            return super().__call__(environ, partial(_start_response_xorigin, start_response))
        return super().__call__(environ, start_response)


def _start_response_xorigin(start_response, status, headers):
    headers_inner = list(headers)
    headers_inner.append(('Access-Control-Allow-Origin', '*'))
    start_response(status, headers_inner)


_CONTENT_TYPES = {
    '.css': 'text/css; charset=utf-8',
    '.js': 'text/javascript; charset=utf-8',
    '.html': 'text/html; charset=utf-8'
}


# Keys that are persisted to the settings file (always saved, regardless of noSave)
_SETTINGS_KEYS = ('model', 'modelOptions', 'templates')


# The ollama-chat configuration context manager
class ConfigManager:
    __slots__ = ('config_path', 'settings_path', 'config_lock', 'config')


    def __init__(self, config_path):
        self.config_path = config_path
        self.config_lock = threading.Lock()

        # Derive the settings file path (same dir, "-settings" suffix)
        base, ext = os.path.splitext(config_path)
        self.settings_path = base + '-settings' + ext

        # Load main config
        if os.path.isfile(self.config_path):
            with open(self.config_path, 'r', encoding='utf-8') as fh_config:
                self.config = schema_markdown.validate_type(OLLAMA_CHAT_TYPES, 'OllamaChatConfig', json.loads(fh_config.read()))
        else:
            self.config = {'conversations': []}

        # Load settings file and apply with highest priority
        if os.path.isfile(self.settings_path):
            try:
                with open(self.settings_path, 'r', encoding='utf-8') as fh_settings:
                    settings = json.loads(fh_settings.read())
                for key in _SETTINGS_KEYS:
                    if key in settings:
                        self.config[key] = settings[key]
            except Exception:  # pragma: no cover
                pass


    @contextmanager
    def __call__(self, save=False):
        # Acquire the config lock
        self.config_lock.acquire()

        try:
            # Yield the config on context entry
            yield self.config

            if save:
                # Always save model options, model name and templates to settings file
                settings = {k: self.config[k] for k in _SETTINGS_KEYS if k in self.config}
                with open(self.settings_path, 'w', encoding='utf-8') as fh_settings:
                    json.dump(settings, fh_settings, indent=4, sort_keys=True)

                # Save full config only when noSave is not set
                if not self.config.get('noSave'):
                    with open(self.config_path, 'w', encoding='utf-8') as fh_config:
                        json.dump(self.config, fh_config, indent=4, sort_keys=True)
        finally:
            # Release the config lock
            self.config_lock.release()


# The model download manager class
class DownloadManager():
    __slots__ = ('app', 'model', 'status', 'completed', 'total', 'stop')


    def __init__(self, app, model):
        self.app = app
        self.model = model
        self.status = ''
        self.completed = 0
        self.total = 0
        self.stop = False

        # Start the download thread
        download_thread = threading.Thread(target=self.download_thread_fn, args=(self, app.pool_manager))
        download_thread.daemon = True
        download_thread.start()


    @staticmethod
    def download_thread_fn(manager, pool_manager):
        try:
            for progress in ollama_pull(pool_manager, manager.model):
                # Stopped?
                if manager.stop:
                    break

                # Update the download status
                manager.status = progress['status']
                manager.completed = progress.get('completed', 0)
                manager.total = progress.get('total')

        except:
            pass

        # Delete the application's download entry
        if manager.model in manager.app.downloads:
            del manager.app.downloads[manager.model]


# The Ollama Chat API type model
with importlib.resources.files('ollama_chat.static').joinpath('ollamaChat.smd').open('r') as cm_smd:
    OLLAMA_CHAT_TYPES = schema_markdown.parse_schema_markdown(cm_smd.read())


@chisel.action(name='getConversations', types=OLLAMA_CHAT_TYPES)
def get_conversations(ctx, unused_req):
    with ctx.app.config() as config:
        response = {
            'conversations': [
                {
                    'id': conversation['id'],
                    'model': conversation['model'],
                    'title': conversation['title'],
                    'generating': conversation['id'] in ctx.app.chats
                }
                for conversation in config['conversations']
            ],
            'templates': [] if 'templates' not in config else [
                {
                    'id': template['id'],
                    'title': template['title']
                }
                for template in config['templates']
            ]
        }
        if 'model' in config:
            response['model'] = config['model']
        if 'modelOptions' in config:
            response['modelOptions'] = config['modelOptions']
        return response


@chisel.action(name='translateText', types=OLLAMA_CHAT_TYPES)
def translate_text(ctx, req):
    from .ollama import _get_ollama_url
    with ctx.app.config() as config:
        model = config.get('model')
    if not model:
        raise chisel.ActionError('NoModel')
    text = req['text']
    from_lang = req['fromLang']
    to_lang = req['toLang']
    level = req.get('level')
    style = req.get('style')
    tone = req.get('tone')
    prompt = f'Translate the following text from {from_lang} to {to_lang}.'
    if level:
        prompt += f' Use {level} language level (CEFR scale).'
    if style:
        prompt += f' Style: {style}.'
    if tone:
        prompt += f' Tone: {tone}.'
    prompt += '\nOutput ONLY the translation, no explanations:\n\n' + text
    url = _get_ollama_url('/api/chat')
    data = {
        'model': model,
        'messages': [{'role': 'user', 'content': prompt}],
        'stream': False
    }
    response = ctx.app.pool_manager.request('POST', url, json=data, retries=0)
    try:
        if response.status != 200:
            raise chisel.ActionError('TranslationFailed')
        result = response.json()
        translation = result.get('message', {}).get('content', '').strip()
    finally:
        response.close()
    return {'translation': translation}


@chisel.action(name='checkGrammar', types=OLLAMA_CHAT_TYPES)
def check_grammar(ctx, req):
    import json as _json
    from .ollama import _get_ollama_url
    with ctx.app.config() as config:
        model = config.get('model')
    if not model:
        raise chisel.ActionError('NoModel')

    text = req['text']
    text_lang = req.get('textLanguage') or 'auto'
    hint_lang = req.get('hintLanguage') or 'auto'

    lang_instr = f'The text is written in: {text_lang}.' if text_lang != 'auto' else 'Detect the text language automatically.'
    hint_instr = f'Write all explanations in {hint_lang}.' if hint_lang != 'auto' else 'Write explanations in the same language as the text.'

    prompt = (
        f'You are a professional proofreader. Fix grammar, spelling, and punctuation.\n'
        f'{lang_instr}\n{hint_instr}\n\n'
        f'Return ONLY a JSON object with this exact structure:\n'
        f'{{"detectedLanguage":"English","corrected":"full corrected text","changes":'
        f'[{{"original":"wrong","corrected":"right","explanation":"reason"}}]}}\n\n'
        f'If no errors found, return changes as []. Do not add any text outside the JSON.\n\n'
        f'Text to proofread:\n{text}'
    )

    url = _get_ollama_url('/api/chat')
    data = {'model': model, 'messages': [{'role': 'user', 'content': prompt}],
            'stream': False, 'format': 'json'}
    response = ctx.app.pool_manager.request('POST', url, json=data, retries=0)
    try:
        if response.status != 200:
            raise chisel.ActionError('CheckFailed', f'Model request failed ({response.status})')
        result = response.json()
        content = result.get('message', {}).get('content', '{}')
    finally:
        response.close()

    try:
        parsed = _json.loads(content)
    except (ValueError, TypeError):
        # Try to extract JSON from response
        import re
        m = re.search(r'\{.*\}', content, re.DOTALL)
        if m:
            try:
                parsed = _json.loads(m.group())
            except (ValueError, TypeError):
                raise chisel.ActionError('CheckFailed', 'Model returned invalid JSON')
        else:
            raise chisel.ActionError('CheckFailed', 'Model returned invalid JSON')

    changes = [
        {'original': c.get('original', ''), 'corrected': c.get('corrected', ''),
         'explanation': c.get('explanation', '')}
        for c in parsed.get('changes', [])
        if c.get('original') and c.get('corrected') and c['original'] != c['corrected']
    ]
    return {
        'detectedLanguage': parsed.get('detectedLanguage', '?'),
        'correctedText': parsed.get('corrected', text),
        'changes': changes
    }


@chisel.action(name='executeCurl', types=OLLAMA_CHAT_TYPES)
def execute_curl(ctx, req):
    import re
    cmd = req['command'].strip()
    # Remove leading 'curl' token and backslash line-continuations
    cmd = re.sub(r'^curl\s+', '', cmd)
    cmd = cmd.replace('\\\n', ' ').replace('\\\r\n', ' ')
    # Parse: extract URL, method, headers, body
    method = 'GET'
    headers = {}
    body = None
    url = None
    i, tokens = 0, _curl_tokenize(cmd)
    while i < len(tokens):
        tok = tokens[i]
        if tok in ('-X', '--request') and i + 1 < len(tokens):
            method = tokens[i + 1]; i += 2
        elif tok in ('-H', '--header') and i + 1 < len(tokens):
            hdr = tokens[i + 1]
            if ':' in hdr:
                k, v = hdr.split(':', 1)
                headers[k.strip()] = v.strip()
            i += 2
        elif tok in ('-d', '--data', '--data-raw', '--data-binary') and i + 1 < len(tokens):
            body = tokens[i + 1]; method = method if method != 'GET' else 'POST'; i += 2
        elif tok in ('--location', '-L', '--compressed', '--silent', '-s', '--verbose', '-v'):
            i += 1
        elif not tok.startswith('-') and url is None:
            url = tok; i += 1
        else:
            i += 1
    if not url:
        raise chisel.ActionError('CurlFailed', 'No URL found in curl command')
    kwargs = {'headers': headers, 'retries': False}
    if body:
        kwargs['body'] = body.encode('utf-8') if isinstance(body, str) else body
    try:
        response = ctx.app.pool_manager.request(method, url, **kwargs)
        try:
            if response.status >= 400:
                raise chisel.ActionError('CurlFailed', f'HTTP {response.status} from {url}')
            content = response.data.decode('utf-8', errors='replace')
        finally:
            response.close()
    except chisel.ActionError:
        raise
    except Exception as exc:
        raise chisel.ActionError('CurlFailed', str(exc))
    return {'content': f'<curl response from {url}>\n{content}\n</curl response>'}


def _curl_tokenize(s):
    """Split curl command into tokens respecting quotes."""
    import shlex
    try:
        return shlex.split(s)
    except ValueError:
        return s.split()


@chisel.action(name='readDirectory', types=OLLAMA_CHAT_TYPES)
def read_directory(ctx, req):
    import pathlib, os
    dir_path = str(pathlib.Path(pathlib.PurePosixPath(req['path'])))
    if not os.path.isdir(dir_path):
        raise chisel.ActionError('DirectoryNotFound', f'Directory not found: {dir_path}')
    ext = req.get('ext', '')
    depth = req.get('depth', 1)
    file_exts = {(ext if ext.startswith('.') else f'.{ext}')} if ext else set()
    from .chat import _get_directory_files, _command_file_content
    parts = []
    try:
        for file_name in sorted(_get_directory_files(dir_path, max(0, depth - 1), file_exts) if file_exts else _get_directory_files(dir_path, max(0, depth - 1), None)):
            file_posix = pathlib.Path(file_name).as_posix()
            try:
                with open(file_name, 'r', encoding='utf-8') as fh:
                    parts.append(_command_file_content(file_posix, fh.read(), False))
            except Exception:
                pass
    except Exception as exc:
        raise chisel.ActionError('DirectoryNotFound', str(exc))
    if not parts:
        raise chisel.ActionError('DirectoryNotFound', f'No files found in {dir_path}')
    return {'content': '\n\n'.join(parts)}


@chisel.action(name='fetchUrl', types=OLLAMA_CHAT_TYPES)
def fetch_url(ctx, req):
    import urllib3
    url = req['url']
    try:
        response = ctx.app.pool_manager.request('GET', url, retries=0)
        try:
            if response.status != 200:
                raise chisel.ActionError('FetchFailed', f'HTTP {response.status} fetching {url}')
            text = response.data.decode('utf-8')
        finally:
            response.close()
    except chisel.ActionError:
        raise
    except Exception as exc:
        raise chisel.ActionError('FetchFailed', str(exc))
    return {'content': f'<{url}>\n{text}\n</{url}>'}


@chisel.action(name='clearModel', types=OLLAMA_CHAT_TYPES)
def clear_model(ctx, unused_req):
    with ctx.app.config(save=True) as config:
        config.pop('model', None)


@chisel.action(name='getModelInfo', types=OLLAMA_CHAT_TYPES)
def get_model_info(ctx, req):
    from .ollama import _get_ollama_url
    model = req['model']
    url_show = _get_ollama_url('/api/show')
    response_show = ctx.app.pool_manager.request('POST', url_show, json={'model': model}, retries=0)
    try:
        if response_show.status != 200:
            raise chisel.ActionError('UnknownModel', f'Unknown model "{model}"')
        model_show = response_show.json()
    finally:
        response_show.close()
    # Read configured num_ctx from model's parameters text (explicit Modelfile override)
    context_length = None
    for line in model_show.get('parameters', '').splitlines():
        parts = line.strip().split()
        if len(parts) >= 2 and parts[0] == 'num_ctx':
            try:
                context_length = int(parts[1])
            except ValueError:
                pass
            break

    # Read architectural context length from model_info (e.g. qwen3.context_length, llama.context_length)
    arch_context_length = None
    for key, value in model_show.get('model_info', {}).items():
        if key.endswith('.context_length'):
            try:
                arch_context_length = int(value)
            except (ValueError, TypeError):
                pass
            break

    # Effective context: explicit param > architecture value
    effective = context_length if context_length is not None else arch_context_length

    result = {'contextLength': effective if effective is not None else 2048}
    if arch_context_length is not None:
        result['maxContextLength'] = arch_context_length
    return result


@chisel.action(name='setModel', types=OLLAMA_CHAT_TYPES)
def set_model(ctx, req):
    with ctx.app.config(save=True) as config:
        config['model'] = req['model']


@chisel.action(name='setModelOptions', types=OLLAMA_CHAT_TYPES)
def set_model_options(ctx, req):
    with ctx.app.config(save=True) as config:
        options = req.get('options') or {}
        if options:
            config['modelOptions'] = options
        elif 'modelOptions' in config:
            del config['modelOptions']


@chisel.action(name='moveConversation', types=OLLAMA_CHAT_TYPES)
def move_conversation(ctx, req):
    with ctx.app.config(save=True) as config:
        # Find the conversation index
        id_ = req['id']
        conversations = config['conversations']
        ix_conv = next((ix for ix, conv in enumerate(conversations) if conv['id'] == id_), None)
        if ix_conv is None:
            raise chisel.ActionError('UnknownConversationID')
        conversation = conversations[ix_conv]

        # Move down?
        if req['down']:
            if ix_conv < len(conversations) - 1:
                conversations[ix_conv] = conversations[ix_conv + 1]
                conversations[ix_conv + 1] = conversation
        else:
            if ix_conv > 0:
                conversations[ix_conv] = conversations[ix_conv - 1]
                conversations[ix_conv - 1] = conversation


@chisel.action(name='moveTemplate', types=OLLAMA_CHAT_TYPES)
def move_template(ctx, req):
    with ctx.app.config(save=True) as config:
        # Find the template index
        id_ = req['id']
        templates = config['templates'] or []
        ix_tmpl = next((ix for ix, tmpl in enumerate(templates) if tmpl['id'] == id_), None)
        if ix_tmpl is None:
            raise chisel.ActionError('UnknownTemplateID')
        template = templates[ix_tmpl]

        # Move down?
        if req['down']:
            if ix_tmpl < len(templates) - 1:
                templates[ix_tmpl] = templates[ix_tmpl + 1]
                templates[ix_tmpl + 1] = template
        else:
            if ix_tmpl > 0:
                templates[ix_tmpl] = templates[ix_tmpl - 1]
                templates[ix_tmpl - 1] = template


@chisel.action(name='deleteTemplate', types=OLLAMA_CHAT_TYPES)
def delete_template(ctx, req):
    with ctx.app.config(save=True) as config:
        id_ = req['id']
        templates = config.get('templates') or []
        ix_tmpl = next((ix for ix, tmpl in enumerate(templates) if tmpl['id'] == id_), None)
        if ix_tmpl is None:
            raise chisel.ActionError('UnknownTemplateID')
        del templates[ix_tmpl]


@chisel.action(name='getTemplate', types=OLLAMA_CHAT_TYPES)
def get_template(ctx, req):
    template_id = req['id']
    with ctx.app.config(save=True) as config:
        templates = config.get('templates') or []
        template = next((template for template in templates if template['id'] == template_id), None)
        if template is None:
            raise chisel.ActionError('UnknownTemplateID')
        return copy.deepcopy(template)


@chisel.action(name='updateTemplate', types=OLLAMA_CHAT_TYPES)
def update_template(ctx, req):
    template_id = req['id']
    with ctx.app.config(save=True) as config:
        templates = config.get('templates') or []
        ix_template = next((ix_tmpl for ix_tmpl, tmpl in enumerate(templates) if tmpl['id'] == template_id), None)
        if ix_template is None:
            raise chisel.ActionError('UnknownTemplateID')
        templates[ix_template] = req


@chisel.action(name='startConversation', types=OLLAMA_CHAT_TYPES)
def start_conversation(ctx, req):
    with ctx.app.config() as config:
        # Compute the conversation title
        user_prompt = req['user']
        max_title_len = 50
        title = re.sub(r'\s+', ' ', user_prompt).strip()
        if len(title) > max_title_len:
            title_suffix = '...'
            title = f'{title[:max_title_len - len(title_suffix)]}{title_suffix}'

        # Create the new conversation object
        model = req.get('model', config.get('model'))
        if model is None:
            raise chisel.ActionError('NoModel')
        id_ = str(uuid.uuid4())
        conversation = {'id': id_, 'model': model, 'title': title, 'exchanges': []}

        # Add the new conversation to the application config
        config['conversations'].insert(0, conversation)

        # Start the model chat
        ctx.app.chats[id_] = ChatManager(ctx.app, id_, [user_prompt], images=req.get('images'))

        # Return the new conversation identifier
        return {'id': id_}


@chisel.action(name='startTemplate', types=OLLAMA_CHAT_TYPES)
def start_template(ctx, req):
    template_id = req['id']
    variable_values = req.get('variables') or {}

    with ctx.app.config() as config:
        # Get the conversation template
        templates = config.get('templates') or []
        template = next((template for template in templates if template['id'] == template_id), None)
        if template is None:
            template = next((template for template in templates if template.get('name') == template_id), None)
        if template is None:
            raise chisel.ActionError('UnknownTemplateID', f'Unknown template "{template_id}"')

        # Get the template prompts
        try:
            title, prompts = config_template_prompts(template, variable_values)
        except ValueError as exc:
            message = str(exc)
            error = 'UnknownVariable' if message.startswith('unknown') else 'MissingVariable'
            raise chisel.ActionError(error, message)

        # Create the new conversation object
        model = req.get('model', config.get('model'))
        if model is None:
            raise chisel.ActionError('NoModel')
        id_ = str(uuid.uuid4())
        conversation = {'id': id_, 'model': model, 'title': title, 'exchanges': []}

        # Add the new conversation to the application config
        config['conversations'].insert(0, conversation)

        # Start the model chat
        ctx.app.chats[id_] = ChatManager(ctx.app, id_, prompts, images=None)

        # Return the new conversation identifier
        return {'id': id_}


@chisel.action(name='stopConversation', types=OLLAMA_CHAT_TYPES)
def stop_conversation(ctx, req):
    with ctx.app.config() as config:
        id_ = req['id']
        conversation = config_conversation(config, id_)
        if conversation is None:
            raise chisel.ActionError('UnknownConversationID')

        # Not generating?
        chat = ctx.app.chats.get(id_)
        if chat is None:
            return

        # Stop the conversation
        chat.stop = True
        del ctx.app.chats[id_]


@chisel.action(name='getConversation', types=OLLAMA_CHAT_TYPES)
def get_conversation(ctx, req):
    with ctx.app.config() as config:
        id_ = req['id']
        conversation = config_conversation(config, id_)
        if conversation is None:
            raise chisel.ActionError('UnknownConversationID')

        # Add the generating status
        conversation = copy.deepcopy(conversation)
        conversation['generating'] = id_ in ctx.app.chats

        # Return the conversation
        return {
            'conversation': conversation
        }


@chisel.action(name='replyConversation', types=OLLAMA_CHAT_TYPES)
def reply_conversation(ctx, req):
    with ctx.app.config() as config:
        id_ = req['id']
        conversation = config_conversation(config, id_)
        if conversation is None:
            raise chisel.ActionError('UnknownConversationID')

        # Busy?
        if id_ in ctx.app.chats:
            raise chisel.ActionError('ConversationBusy')

        # Start the model chat
        ctx.app.chats[id_] = ChatManager(ctx.app, id_, [req['user']], images=req.get('images'))


@chisel.action(name='setConversationThinking', types=OLLAMA_CHAT_TYPES)
def set_conversation_thinking(ctx, req):
    with ctx.app.config(save=True) as config:
        id_ = req['id']
        conversation = config_conversation(config, id_)
        if conversation is None:
            raise chisel.ActionError('UnknownConversationID')
        if id_ in ctx.app.chats:
            raise chisel.ActionError('ConversationBusy')
        enabled = req.get('enabled')
        if enabled is None:
            conversation.pop('thinkingEnabled', None)
        else:
            conversation['thinkingEnabled'] = enabled


@chisel.action(name='setConversationTitle', types=OLLAMA_CHAT_TYPES)
def set_conversation_title(ctx, req):
    with ctx.app.config(save=True) as config:
        id_ = req['id']
        conversation = config_conversation(config, id_)
        if conversation is None:
            raise chisel.ActionError('UnknownConversationID')

        # Busy?
        if id_ in ctx.app.chats:
            raise chisel.ActionError('ConversationBusy')

        # Set the conversation title
        conversation['title'] = req['title']


@chisel.action(name='deleteConversation', types=OLLAMA_CHAT_TYPES)
def delete_conversation(ctx, req):
    with ctx.app.config(save=True) as config:
        id_ = req['id']
        conversation = config_conversation(config, id_)
        if conversation is None:
            raise chisel.ActionError('UnknownConversationID')

        # Busy?
        if id_ in ctx.app.chats:
            raise chisel.ActionError('ConversationBusy')

        # Delete the conversation
        config['conversations'] = [conversation for conversation in config['conversations'] if conversation['id'] != id_]


@chisel.action(name='createTemplate', types=OLLAMA_CHAT_TYPES)
def create_template(ctx, req):
    with ctx.app.config(save=True) as config:
        # Create the new template
        id_ = str(uuid.uuid4())
        template = {
            'id': id_,
            'title': req['title'],
            'prompts': req['prompts']
        }
        if 'name' in req:
            template['name'] = req['name']
        if 'variables' in req:
            template['variables'] = req['variables']

        # Add the new template to the application config
        if 'templates' not in config:
            config['templates'] = []
        config['templates'].insert(0, template)

        # Return the new template identifier
        return {'id': id_}


@chisel.action(name='deleteConversationExchange', types=OLLAMA_CHAT_TYPES)
def delete_conversation_exchange(ctx, req):
    with ctx.app.config(save=True) as config:
        id_ = req['id']
        conversation = config_conversation(config, id_)
        if conversation is None:
            raise chisel.ActionError('UnknownConversationID')

        # Busy?
        if id_ in ctx.app.chats:
            raise chisel.ActionError('ConversationBusy')

        # Delete the most recent exchange
        exchanges = conversation['exchanges']
        if len(exchanges):
            del exchanges[-1]


@chisel.action(name='regenerateConversationExchange', types=OLLAMA_CHAT_TYPES)
def regenerate_conversation_exchange(ctx, req):
    with ctx.app.config(save=True) as config:
        id_ = req['id']
        conversation = config_conversation(config, id_)
        if conversation is None:
            raise chisel.ActionError('UnknownConversationID')

        # Busy?
        if id_ in ctx.app.chats:
            raise chisel.ActionError('ConversationBusy')

        # Any exchanges?
        exchanges = conversation['exchanges']
        if len(exchanges):
            # Delete the most recent exchange
            prompt = exchanges[-1]['user']
            del exchanges[-1]

            # Start the model chat
            ctx.app.chats[id_] = ChatManager(ctx.app, id_, [prompt], images=None)


@chisel.action(name='getModels', types=OLLAMA_CHAT_TYPES)
def get_models(ctx, unused_req):
    # Get the Ollama models
    models = ollama_list(ctx.app.pool_manager)

    # Create the models response
    response_models = [
        {
            'id': model['model'],
            'name': model['model'][:model['model'].index(':')],
            'parameters': _parse_parameter_size(ctx, model['details']['parameter_size']),
            'size': model['size'],
            'modified': model['modified_at']
        }
        for model in models
    ]

    with ctx.app.config() as config:
        # Create the downloading models response
        downloading_models = []
        for model_id, download_manager in ctx.app.downloads.items():
            download = {
                'id': model_id,
                'status': download_manager.status,
                'completed': download_manager.completed
            }
            if download_manager.total:
                download['size'] = download_manager.total
            downloading_models.append(download)

        response = {
            'models': sorted(response_models, key=lambda model: model['id']),
            'downloading': sorted(downloading_models, key=lambda model: model['id'])
        }
        if 'model' in config:
            response['model'] = config['model']
        return response


def _parse_parameter_size(ctx, parameter_size):
    value = float(parameter_size[:-1])
    unit = parameter_size[-1]
    if unit == 'B':
        return int(value * 1000000000)
    elif unit == 'M':
        return int(value * 1000000)
    elif unit == 'K':
        return int(value * 1000)

    ctx.log.warning(f'Invalid parameter size "{parameter_size}"')
    return 0


@chisel.action(name='downloadModel', types=OLLAMA_CHAT_TYPES)
def download_model(ctx, req):
    with ctx.app.config():
        ctx.app.downloads[req['model']] = DownloadManager(ctx.app, req['model'])


@chisel.action(name='stopModelDownload', types=OLLAMA_CHAT_TYPES)
def stop_model_download(ctx, req):
    with ctx.app.config():
        if req['model'] in ctx.app.downloads:
            ctx.app.downloads[req['model']].stop = True


@chisel.action(name='deleteModel', types=OLLAMA_CHAT_TYPES)
def delete_model(ctx, req):
    ollama_delete(ctx.app.pool_manager, req['model'])


@chisel.action(name='getSystemInfo', types=OLLAMA_CHAT_TYPES)
def get_system_info(unused_ctx, unused_req):
    # Compute the total memory
    if platform.system() == "Windows": # pragma: no cover
        memory_status = MEMORYSTATUSEX()
        # pylint: disable-next=invalid-name, attribute-defined-outside-init
        memory_status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory_status))
        total_memory = memory_status.ullTotalPhys
    else: # pragma: no cover
        # pylint: disable-next=no-member, useless-suppression
        total_memory = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")

    return {
        'memory': total_memory
    }


class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_uint),
        ("dwMemoryLoad", ctypes.c_uint),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]

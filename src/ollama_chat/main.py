# Licensed under the MIT License
# https://github.com/craigahobbs/ollama-chat/blob/main/LICENSE

"""
ollama-chat command-line script main module
"""

import argparse
import os
import sys
import threading
import webbrowser

import waitress

from .app import OllamaChat


# The default config file name
CONFIG_FILENAME = 'ollama-chat.json'


def main(argv=None):
    """
    ollama-chat command-line script main entry point
    """

    # Command line arguments
    argument_parser_args = {'prog': 'ollama-chat'}
    if sys.version_info >= (3, 14): # pragma: no cover
        argument_parser_args['color'] = False
    parser = argparse.ArgumentParser(**argument_parser_args)
    parser.add_argument('-c', metavar='FILE', dest='config',
                        help='the configuration file (default is "$HOME/ollama-chat.json")')
    parser.add_argument('-p', metavar='N', dest='port', type=int, default=8080,
                        help='the application port (default is 8080)')
    parser.add_argument('-x', dest='xorigin', action='store_true', default=False,
                        help="enable cross-origin back-end requests")
    parser.add_argument('-b', dest='backend', action='store_false', default=True,
                        help="don't start the back-end (use existing)")
    parser.add_argument('-n', dest='browser', action='store_false', default=True,
                        help="don't open a web browser")
    parser.add_argument('-q', dest='quiet', action='store_true',
                        help="don't display access logging")
    args = parser.parse_args(args=argv)

    # Starting a backend server? If so, create the backend application.
    if args.backend:

        # Determine the config path
        config_path = args.config
        if config_path is None:
            if os.path.isfile(CONFIG_FILENAME):
                config_path = CONFIG_FILENAME
            else:
                config_path = os.path.join(os.path.expanduser('~'), CONFIG_FILENAME)
        elif config_path.endswith(os.sep) or os.path.isdir(config_path):
            config_path = os.path.join(config_path, CONFIG_FILENAME)

        # Create the backend application
        application = OllamaChat(config_path, args.xorigin)

    # Construct the URL
    host = '127.0.0.1'
    url = f'http://{host}:{args.port}/'

    # Launch the web browser on a thread (it may block)
    if args.browser:
        webbrowser_thread = threading.Thread(target=webbrowser.open, args=(url,))
        webbrowser_thread.daemon = True
        webbrowser_thread.start()

    # Host the application
    if args.backend:

        # Wrap the backend so we can log status and environ
        def application_wrap(environ, start_response):
            def log_start_response(status, response_headers):
                if not args.quiet:
                    print(f'ollama-chat: {status[0:3]} {environ["REQUEST_METHOD"]} {environ["PATH_INFO"]} {environ["QUERY_STRING"]}')
                return start_response(status, response_headers)
            return application(environ, log_start_response)

        # Start the backend application
        if not args.quiet:
            print(f'ollama-chat: Serving at {url} ...')
        waitress.serve(application_wrap, port=args.port)

    # Not starting a backend service, so we must wait on the web browser start
    elif args.browser:
        webbrowser_thread.join()

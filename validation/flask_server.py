from flask import Flask, jsonify, request
import logging
import os
import signal

def create_flask_app(agent, port):
    app = Flask(__name__)
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)

    @app.route('/select_action', methods=['POST'])
    def select_action():
        data = request.get_json()
        state = data.get('state') 
        action = agent.act(state)
        return jsonify({'action': action}), 200
    
    @app.route('/shutdown', methods=['POST'])
    def shutdown():
        os.kill(os.getpid(), signal.SIGTERM)
        return 'Server shutting down...'

    return app

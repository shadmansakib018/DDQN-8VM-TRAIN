from flask import Flask, jsonify, request
from threading import Thread
import logging

def create_flask_app(agent, port):
    app = Flask(__name__)
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)

    @app.route('/fallback_store_states', methods=['POST'])
    def fallback_store_states():
        data = request.get_json()
        state = data.get('state')
        action = data.get('action') 
        reward = data.get('reward')
        next_state = data.get('next_state')
        source_port = data.get('port')
        agent.remember(state, action, reward, next_state, source_port)
        return jsonify({'success': True, 'message': 'Training step completed'}), 200
    

    @app.route('/store_states', methods=['POST'])
    def store_states():
        data = request.get_json()
        state = data.get('state')
        action = data.get('action') 
        reward = data.get('reward')
        next_state = data.get('next_state')
        agent.remember(state, action, reward, next_state, port)
        return jsonify({'success': True, 'message': 'Training step completed'}), 200

    @app.route('/train_now', methods=['POST'])
    def trigger_train():
        def run_training():
            print("training started")
            agent.train(port)
            print("training finished")
        Thread(target=run_training).start()
        return jsonify({"status": "training_started"})

    @app.route('/select_action', methods=['POST'])
    def select_action():
        data = request.get_json()
        state = data.get('state') 
        action = agent.act(state)
        return jsonify({'action': action}), 200

    @app.route('/long_term_reward', methods=['POST'])
    def long_term_reward():
        data = request.get_json()
        avg_response_time = data.get('avg_response_time')

        if avg_response_time is not None:
            with open("long_term_rewards.txt", "a") as f:
                f.write(f"{avg_response_time:.2f}\n")
            # print(f"Received and logged long term reward: {avg_response_time:.2f}")
            return jsonify({'success': True, 'message': 'Long term reward recorded'}), 200
        else:
            return jsonify({'success': False, 'message': 'Invalid data received'}), 400

    return app

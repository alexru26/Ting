import json
import sys

from state import GameState
from policy import create_policy


class MahjongBot:
    """
    Botzone-compatible entry point.

    Each turn Botzone sends one JSON object on stdin:
        {"requests": [...all requests so far...], "responses": [...all previous responses...]}
    The bot writes one JSON object to stdout:
        {"response": "<action>"}

    The neural model is the only decision source. Legality is guaranteed by
    construction: the policy chooses from GameState.enumerate_legal_actions(),
    which enumerates exactly the moves Botzone will accept. Malformed input
    or a missing/incompatible checkpoint raises instead of being silently
    papered over.
    """

    def handle_input(self, payload):
        requests = payload['requests']
        responses = payload.get('responses', [])
        if not requests:
            raise ValueError('Empty request history')

        state = GameState.from_history(requests, responses)
        if state.last_request_type in (0, 1):
            # Seat/deal announcements have no decision; the protocol answer is PASS.
            return 'PASS'
        if state.last_request_type not in (2, 3):
            raise ValueError('Unrecognized request: %r' % (requests[-1],))

        policy = create_policy(state)
        return policy.choose_action()

    def run(self):
        payload = json.loads(sys.stdin.read())
        response = self.handle_input(payload)
        print(json.dumps({'response': response}))


if __name__ == '__main__':
    MahjongBot().run()

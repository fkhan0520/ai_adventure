import os
import openai
from typing import List, Tuple
import numpy as np
from sklearn_extra.cluster import KMedoids
import time
from loguru import logger
from threading import Thread, Lock
from twilio.rest import Client
from pathlib import Path

openai.api_key = os.getenv("OPENAI_API_KEY")

CONTEXT_LIMIT = 4096
INITIAL_GAME_TEXT = """
Welcome to FAN Club's text based adventure game! We will all be playing a game together in real time. We will be put in a scenario starting in this apartment and we will all suggest a course of action. An AI will gather all our suggestions, select the most representative suggestion, and continue the adventure. Every 10 minutes, the AI will update everyone with the next step in the adventure. Feel free to be as creative as you want! (BUT RESPONSES ARE NOT ANONYMOUS) Here is the scenario...
"""
ACTION_RECEIVED_TEXT = "Your input has been received! Waiting for all players..."


def get_initial_prompt():
    return INITIAL_GAME_TEXT


def send_twilio_message(receiver: str, message: str):
    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")
    client = Client(account_sid, auth_token)
    message = client.messages.create(to=receiver, from_=os.getenv("TWILIO_PHONE_NUMBER"), body=message)
    logger.debug(f"Message sent to {receiver}, {message.sid}")


class AdventureContext:
    """
    State management for the Adventure game.
    """

    def __init__(self, context_name: str):
        self.context_name = context_name
        with open(self.prompt_file, "r") as f:
            self.initial_prompt = f.read()
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "user",
                    "content": self.initial_prompt,
                }
            ],
        )
        first_step = completion.choices[0].message.content
        with open(self.context_file, "w") as f:
            f.write(self.initial_prompt + "\n")
            f.write(first_step + "\n")
        self.action_number = 0
        self.daemon = Thread(target=self._run_thread)
        self.lock = Lock()  # probably not needed but want to avoid file i/o weirdness
        Path(self.users_file).touch(exist_ok=True)
        Path(self.responses_file).touch(exist_ok=True)
        self.game_started = False
        logger.info(f"{self.context_name} initialized")

    def run_game_loop(self):
        try:
            self.daemon.start()
            self.game_started = True
        except Exception as e:
            logger.error(f"Error starting thread: {repr(e)}")

    def receive_message(self, phone_number: str, message: str) -> List[str]:
        with self.lock:
            user_exists = self._maybe_register_user(phone_number, message)
            if not user_exists:
                return [INITIAL_GAME_TEXT] + self._get_history()[1:]
            self._collect_action(phone_number, message)
            return [ACTION_RECEIVED_TEXT]

    def _maybe_register_user(self, phone_number: str, message: str) -> bool:
        user_exists = False
        with open(self.users_file, "r") as f:
            for line in f:
                if line.strip().split("|")[0] == phone_number:
                    user_exists = True
        if not user_exists:
            with open(self.users_file, "a") as f:
                f.write(f"{phone_number}|{message}\n")
        return user_exists

    def _get_history(self) -> List[str]:
        with open(self.context_file, "r") as f:
            rawlines = f.readlines()
        return [x.strip() for x in rawlines]

    def _collect_action(self, phone_number: str, action: str):
        # TODO: make sure action was not already received
        with open(self.responses_file, "a") as f:
            f.write(f"{phone_number}|{action}\n")

    def _get_embeddings(self, responses: List[str]) -> np.array:
        embeddings = []
        for response in responses:
            embedding = openai.Embedding.create(
                input=[response], model="text-embedding-ada-002"
            )["data"][0]["embedding"]
            embeddings.append(embedding)
        return np.array(embeddings)

    def _vote(self, responses: List[str]) -> int:
        embeddings = self._get_embeddings(responses)
        kmedoids = KMedoids(n_clusters=1, random_state=0).fit(embeddings)
        return kmedoids.medoid_indices_[0]

    def _vote_and_step_game(self) -> str:
        responses = []
        users = []
        with open(self.responses_file, "r") as f:
            for line in f.readlines():
                tokens = line.strip().split("|")
                responses.append(tokens[1])
                users.append(tokens[0])
        voted_response_idx = self._vote(responses)
        voted_response = responses[voted_response_idx]
        voted_user = users[voted_response_idx]
        history = self._get_history()
        messages = []
        # TODO: respect token limit
        for i, message in enumerate(history):
            role = "user" if i % 2 == 0 else "assistant"
            messages.append(
                {
                    "role": role,
                    "content": message,
                }
            )
        messages.append({"role": "user", "content": voted_response})
        logger.debug("Making GPT request....")
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=messages
        )
        logger.debug("Done making GPT request")
        output = completion.choices[0].message.content
        with open(self.context_file, "a") as f:
            f.write(voted_response + "\n")
            f.write(output + "\n")
        self.action_number += 1
        Path(self.responses_file).touch(exist_ok=True)
        for phone_number, name in self._get_users():
            if voted_user == phone_number:
                voted_user = name
                break
        logger.debug(f"Voted user: {voted_user}, action: {voted_response}")
        message = f"The AI has spoken! The next action (submitted by {voted_user}) is:\n\n{voted_response}\n\nThe result of the action is:\n\n{output}"
        return message

    def _get_users(self) -> List[Tuple[str]]:
        with open(self.users_file, "r") as f:
            users = []
            for line in f.readlines():
                tokens = line.strip().split("|")
                users.append((tokens[0], tokens[1]))
            return users

    def _broadcast_players(self, broadcast_message: str):
        for phone_number, _ in self._get_users():
            send_twilio_message(phone_number, broadcast_message)

    def _run_thread(self):
        while True:
            time.sleep(60 * 10)
            with self.lock:
                logger.info("RUNNING NEXT ROUND OF ADVENTURE")
                broadcast_message = self._vote_and_step_game()
                self._broadcast_players(broadcast_message)

    """
    {context_name}.txt: initial prompt
    {context_name}_users.txt: list of phone numbers
    {context_name}_responses_{i}.txt: each user's response to the prompt
    {context_name}_context.txt: message history that is actually sent to GPT
    """

    @property
    def prompt_file(self) -> str:
        return f"{self.context_name}.txt"

    @property
    def context_file(self) -> str:
        return f"{self.context_name}_context.txt"

    @property
    def users_file(self) -> str:
        return f"{self.context_name}_users.txt"

    @property
    def responses_file(self) -> str:
        return f"{self.context_name}_responses_{self.action_number}.txt"

# ai_adventure
A collaborative GPT-based text adventure game for use with large groups.

The game works by first seeding GPT with a prompt to set up an adventure game. A sample prompt is provided in `iftar_prompt.txt`. Then each participant will periodically (default 10 min) receive scenarios and respond with a suggestion for what to do next. The suggestions from all players are then converted into embeddings and "voted" on by selecting the medoid of all the embeddings. The medoid is sent to GPT to drive the next step of the adventure and the process starts all over again.

Requires access to Twilio and OpenAI APIs

## Setting it up:
0. Get a Twilio phone number and OpenAI API credentials

1. Install requirements:
```
pip install -r requirements.txt
```
2. Export the following env vars:
```
export OPENAI_API_KEY=...
export TWILIO_ACCOUNT_SID=...
export TWILIO_AUTH_TOKEN=...
export TWILIO_PHONE_NUMBER=...
```

3. Run the server:
```
uvicorn server:app
```

4. Add `<your_host>:<your_port>/hook` as the webhook for handling SMS on your Twilio number

## Gameplay

Once the server is running, ask all participants to text the `TWILIO_PHONE_NUMBER` with their name or some other identifier. They should each receive a text with some game instructions and the initial prompt. They can begin sending suggestions for the first scenario.

Once everyone has been set up, send a post request to `/start_game` to begin the game loop. Remember that chosen responses are not anonymous :P




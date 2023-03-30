from fastapi import FastAPI, Form, Response
from twilio.twiml.messaging_response import MessagingResponse
from ai import AdventureContext

app = FastAPI()

game_context = AdventureContext("iftar_prompt")


@app.post("/hook")
async def chat(From: str = Form(...), Body: str = Form(...)):
    response = MessagingResponse()
    messages = game_context.receive_message(From, Body)
    for m in messages:
        response.message(m)
    return Response(content=str(response), media_type="application/xml")


@app.post("/start_game")
def start_game():
    game_context.run_game_loop()
    return {"message": "game started"}

from dotenv import load_dotenv
import discord
from discord.ext import commands
from discord import app_commands
import os
from llama_cpp import Llama
import asyncio

# Download model from here: 
# https://huggingface.co/TheBloke/toxicqa-Llama2-13B-GGUF/blob/main/toxicqa-llama2-13b.Q8_0.gguf
LLM = Llama(model_path="./models/toxicqa-llama2-13b.Q8_0.gguf",
            n_ctx=4096,
            n_gpu_layers=5)


intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix=".", intents=intents)
tree = bot.tree

@bot.event
async def on_ready():
    if bot.user != None:
        print("Logged in as", bot.user.name) 
    synced = await bot.tree.sync()
    print(synced)


async def summarize(text) -> str:
    print("summarizing...")
    prompt = "Q: Summarize the following text in one paragraph. Do not repeat my question in the response. For all pronouns use they and them. Do not ask for clarification. Only summarize: " + text + " A:"
    loop = asyncio.get_running_loop()
    output = await loop.run_in_executor(None, lambda: LLM(prompt, max_tokens=0))
    output = output["choices"][0]["text"]
    print(output)
    return output


async def history(ctx: discord.Interaction, amount: int):
    final = ""
    if ctx.channel != None:
        messages = [message async for message in ctx.channel.history(limit=amount)]
        print(messages)
        for message in messages:
            print(message.content)
            final += message.content
    return final



@tree.command(name="tldr", description="Gives TLDR of the last x messages")
async def tldr(interaction: discord.Interaction, number_of_messages: int):
    print("tldr run")
    messages = await history(interaction, number_of_messages)
    await interaction.response.send_message(
            "Summarizing...",
            ephemeral=True
            )
    summary = await summarize(messages)
    original_interaction = await interaction.original_response()
    await original_interaction.edit(
            content=summary
            )
    

def main():

    load_dotenv()
    TOKEN = os.getenv("TOKEN")
    if TOKEN != None:
        bot.run(TOKEN)


if __name__ == "__main__":
    main()

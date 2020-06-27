import discord
import easy_smitebuilds
import os

client = discord.Client()

with open(os.path.join("..", "bot_info.txt")) as infile:
    datapath = infile.readline().strip()
    token = infile.readline().strip()


@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if message.content.startswith("!build"):
        god = message.content.split(" ")[1]
        builds = easy_smitebuilds.main(datapath, "conquest", god, 15, 0.5, 0.7)

        if builds is None:
            await message.channel.send("An error occurred")

        else:
            send_str = ""
            for smitebuild in builds:
                send_str += "dt_rank: " + str(smitebuild.dt_rank) + "\n"
                send_str += "bnb_rank: " + str(smitebuild.bnb_rank) + "\n"
                send_str += "core: " + str(smitebuild.build.core) + "\n"
                send_str += "optional: " + str(smitebuild.build.optional) + "\n"
                send_str += "\n"

            await message.channel.send(send_str)


client.run(token)
